//! NIP-05: mapping nostr keys to DNS-based internet identifiers.
//!
//! Three routes:
//! - `GET /.well-known/nostr.json?name=<local-part>` — public, spec-shaped
//!   lookup. Serves the static overlay first (domain root `_`, official
//!   accounts), then the dynamic registry. Only ever answers the queried
//!   name; unknown names return an uniform 404 so the endpoint doubles as
//!   an existence oracle and nothing more.
//! - `POST /lnurlpay/{pubkey}/nostr` — spark accounts bind their nostr key
//!   via the standard Spark identity-key signature plus a kind-22242 proof
//!   event carrying the account's `lnaddress` tag.
//! - `POST /nostr/blink` — blink (custodial) accounts do the same, authed
//!   by forwarding their Blink session token to the GraphQL `me` query.
//!
//! The proof event is what stops impersonation: registering a nostr pubkey
//! you do not control (i.e. someone else's established nostr identity)
//! fails because the event must be signed by that very key.

use std::collections::HashMap;

use axum::{
    Extension, Json,
    extract::{Path, Query},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
};
use nostr::{Event, JsonUtil, Kind, TagKind};
use serde_json::Value;
use tracing::{debug, trace, warn};

use crate::country::client_ip;
use crate::models::{
    ERROR_RATE_LIMITED, NostrJsonResponse, RegisterBlinkNostrIdentityRequest,
    RegisterNostrIdentityRequest, RegisterNostrIdentityResponse,
};
use crate::repository::{AccountProvider, LnurlRepository};
use crate::routes::{LnurlServer, account};
use crate::state::State;
use crate::time::now_u64;

/// Short cache: verification clients re-fetch rarely, but a short TTL keeps
/// key rotation and handle transfers from lingering after the DB changes.
const NOSTR_JSON_CACHE_CONTROL: &str = "public, max-age=60";
/// nostr kinds accepted as proof-of-key: 22242 (client authentication) and
/// 27235 (website login) — the two kinds the Blink app already produces
/// with an `lnaddress` tag for `BTCPay` provisioning.
const PROOF_KINDS: [Kind; 2] = [Kind::Authentication, Kind::HttpAuth];

impl<DB> LnurlServer<DB>
where
    DB: LnurlRepository + Clone + Send + Sync + 'static,
{
    /// Public NIP-05 lookup: `GET /.well-known/nostr.json?name=<name>`.
    pub async fn handle_nostr_json(
        Query(params): Query<HashMap<String, String>>,
        headers: HeaderMap,
        Extension(state): Extension<State<DB>>,
    ) -> Result<Response, (StatusCode, Json<Value>)> {
        if !state.ip_rate_limiter.check(client_ip(&headers)) {
            return Err((
                StatusCode::TOO_MANY_REQUESTS,
                Json(Value::String(ERROR_RATE_LIMITED.into())),
            ));
        }

        let Some(name) = params.get("name").map(String::as_str) else {
            // The spec always sends `name`; without it there is nothing to
            // answer — and we never enumerate the map.
            return Err(not_found());
        };
        if !valid_nip05_local_part(name) {
            return Err(not_found());
        }

        // Static overlay first: immune to the registration lifecycle and
        // covers service identities (the `_` domain root, official keys).
        if let Some(pubkey) = state.nostr_static_names.get(name) {
            return Ok(nostr_json_response(name, pubkey));
        }

        // The endpoint is Host-scoped like every other public route: the
        // domain of the handle must match the domain being queried.
        let host = headers
            .get(axum::http::header::HOST)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string)
            .unwrap_or_default();
        let domain = account::sanitize_domain(&state, &host).await?;

        match state
            .db
            .get_nostr_identity_by_identifier(&domain, name)
            .await
        {
            Ok(Some(identity)) => Ok(nostr_json_response(name, &identity.nostr_pubkey)),
            Ok(None) => Err(not_found()),
            Err(e) => {
                warn!("nostr.json lookup failed for '{name}': {e}");
                Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(Value::String("internal server error".into())),
                ))
            }
        }
    }

    /// Bind a nostr key to a spark account's handle.
    /// Canonical signed message: `nostr:{nostr_pubkey}-{timestamp}`.
    pub async fn register_nostr(
        Path(pubkey): Path<String>,
        headers: HeaderMap,
        Extension(state): Extension<State<DB>>,
        Json(payload): Json<RegisterNostrIdentityRequest>,
    ) -> Result<Json<RegisterNostrIdentityResponse>, (StatusCode, Json<Value>)> {
        account::require_spark_provider_enabled(&state)?;
        if !state.ip_rate_limiter.check(client_ip(&headers)) {
            return Err((
                StatusCode::TOO_MANY_REQUESTS,
                Json(Value::String(ERROR_RATE_LIMITED.into())),
            ));
        }

        let nostr_pubkey = normalize_nostr_pubkey(&payload.nostr_pubkey)?;
        if payload.nostr_proof.len() > super::MAX_NOSTR_EVENT_SIZE {
            return Err(bad_request("nostr proof too large"));
        }

        let host = host_header(&headers);
        let domain = account::sanitize_domain(&state, &host).await?;

        let pubkey = account::validate(
            &pubkey,
            &payload.signature,
            &format!("nostr:{nostr_pubkey}"),
            payload.timestamp,
            &state,
        )
        .await?;

        let spark_account = state
            .db
            .get_account_by_spark_pubkey(&pubkey.to_string())
            .await
            .map_err(account::storage_error)?;
        let Some(spark_account) = spark_account else {
            return Err((
                StatusCode::NOT_FOUND,
                Json(Value::String(
                    "no account registered for this pubkey".into(),
                )),
            ));
        };

        let username = state
            .db
            .get_spark_username_by_pubkey(&domain, &pubkey.to_string())
            .await
            .map_err(account::storage_error)?
            .map(|u| u.username)
            .ok_or_else(|| {
                (
                    StatusCode::NOT_FOUND,
                    Json(Value::String(
                        "no username registered for this account on this domain".into(),
                    )),
                )
            })?;

        verify_nostr_identity_proof(
            &payload.nostr_proof,
            &nostr_pubkey,
            &format!("{username}@{domain}"),
        )?;

        state
            .db
            .upsert_nostr_identity(&spark_account.account_id, &domain, &nostr_pubkey)
            .await
            .map_err(account::storage_error)?;

        debug!("registered nostr identity {nostr_pubkey} for {username}@{domain}");
        Ok(Json(RegisterNostrIdentityResponse {
            nip05: format!("{username}@{domain}"),
            username,
            domain,
            nostr_pubkey,
        }))
    }

    /// Bind a nostr key to a blink (custodial) account's handle. Auth is a
    /// Blink session token validated server-side via the GraphQL `me`
    /// query; the username must already be provisioned in the registry by
    /// Blink Core (internal route) — this route never creates identifiers.
    pub async fn register_nostr_blink(
        headers: HeaderMap,
        Extension(state): Extension<State<DB>>,
        Json(payload): Json<RegisterBlinkNostrIdentityRequest>,
    ) -> Result<Json<RegisterNostrIdentityResponse>, (StatusCode, Json<Value>)> {
        if !state.ip_rate_limiter.check(client_ip(&headers)) {
            return Err((
                StatusCode::TOO_MANY_REQUESTS,
                Json(Value::String(ERROR_RATE_LIMITED.into())),
            ));
        }

        let nostr_pubkey = normalize_nostr_pubkey(&payload.nostr_pubkey)?;
        if payload.nostr_proof.len() > super::MAX_NOSTR_EVENT_SIZE {
            return Err(bad_request("nostr proof too large"));
        }

        let host = host_header(&headers);
        let domain = account::sanitize_domain(&state, &host).await?;

        let Some(token) = bearer_token(&headers) else {
            return Err((
                StatusCode::UNAUTHORIZED,
                Json(Value::String("missing bearer token".into())),
            ));
        };

        let me = state
            .providers
            .blink_me(&token)
            .await
            .map_err(|e| match e {
                blink_client::BlinkClientError::Graphql(_) => (
                    StatusCode::UNAUTHORIZED,
                    Json(Value::String("invalid or expired token".into())),
                ),
                error => {
                    debug!("blink me lookup failed: {error}");
                    (
                        StatusCode::BAD_GATEWAY,
                        Json(Value::String("blink api unavailable".into())),
                    )
                }
            })?;
        let username = me.username.ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(Value::String("account has no username".into())),
            )
        })?;

        let recipient = state
            .db
            .resolve_recipient_by_identifier(&domain, &username)
            .await
            .map_err(account::storage_error)?;
        let recipient = recipient.ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(Value::String(
                    "username not provisioned on this domain".into(),
                )),
            )
        })?;
        if recipient.provider != AccountProvider::Blink {
            return Err((
                StatusCode::CONFLICT,
                Json(Value::String(
                    "username belongs to a non-blink account".into(),
                )),
            ));
        }
        if recipient.identifier != username {
            return Err((
                StatusCode::CONFLICT,
                Json(Value::String("username mismatch".into())),
            ));
        }

        verify_nostr_identity_proof(
            &payload.nostr_proof,
            &nostr_pubkey,
            &format!("{username}@{domain}"),
        )?;

        state
            .db
            .upsert_nostr_identity(&recipient.account_id, &domain, &nostr_pubkey)
            .await
            .map_err(account::storage_error)?;

        debug!(
            "registered nostr identity {nostr_pubkey} for blink account {} ({username}@{domain})",
            recipient.account_id
        );
        Ok(Json(RegisterNostrIdentityResponse {
            nip05: format!("{username}@{domain}"),
            username,
            domain,
            nostr_pubkey,
        }))
    }
}

/// Verify a kind-22242/27235 proof event:
/// - valid nostr id + signature by the claimed key
/// - exactly one `lnaddress` tag equal to `username@domain`
/// - `created_at` within the acceptable window of now
pub(super) fn verify_nostr_identity_proof(
    nostr_proof: &str,
    nostr_pubkey: &str,
    expected_lnaddress: &str,
) -> Result<(), (StatusCode, Json<Value>)> {
    let event = Event::from_json(nostr_proof).map_err(|e| {
        trace!("invalid nostr proof, could not parse: {e}");
        bad_request("invalid nostr proof")
    })?;

    if !PROOF_KINDS.contains(&event.kind) {
        trace!("nostr proof has unexpected kind: {:?}", event.kind);
        return Err(bad_request("nostr proof has unexpected kind"));
    }

    if event.verify().is_err() {
        trace!("nostr proof does not verify");
        return Err(bad_request("invalid nostr proof"));
    }

    // nostr_pubkey is already validated as canonical lowercase hex.
    if event.pubkey.to_hex() != nostr_pubkey {
        trace!("nostr proof signed by a different key than claimed");
        return Err(bad_request("nostr proof key mismatch"));
    }

    let now = now_u64();
    if event.created_at.as_secs().abs_diff(now) > super::ACCEPTABLE_TIME_DIFF_SECS {
        trace!(
            "nostr proof timestamp too far off: {}, now: {}",
            event.created_at.as_secs(),
            now
        );
        return Err(bad_request("stale nostr proof"));
    }

    let lnaddress: Vec<&str> = event
        .tags
        .iter()
        .filter(|tag| matches!(tag.kind(), TagKind::Custom(kind) if kind == "lnaddress"))
        .filter_map(|tag| tag.content())
        .collect();
    if lnaddress.as_slice() != [expected_lnaddress] {
        trace!(
            "nostr proof lnaddress tag mismatch: expected {expected_lnaddress}, got {lnaddress:?}"
        );
        return Err(bad_request("nostr proof lnaddress mismatch"));
    }

    Ok(())
}

/// NIP-05 local-part charset: `a-z0-9-_.` (lowercase only, per spec).
pub(super) fn valid_nip05_local_part(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name.bytes().all(|b| {
            b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'-' | b'_' | b'.')
        })
}

/// Accept only canonical lowercase 64-char hex x-only pubkeys. Refuse npubs
/// and mixed case: the spec requires lowercase hex in nostr.json.
pub(super) fn normalize_nostr_pubkey(
    nostr_pubkey: &str,
) -> Result<String, (StatusCode, Json<Value>)> {
    let trimmed = nostr_pubkey.trim();
    if trimmed.len() == 64
        && trimmed
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Ok(trimmed.to_string());
    }
    trace!("invalid nostr pubkey format: {trimmed:?}");
    Err(bad_request("invalid nostr pubkey"))
}

fn host_header(headers: &HeaderMap) -> String {
    headers
        .get(axum::http::header::HOST)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_string()
}

fn bearer_token(headers: &HeaderMap) -> Option<String> {
    headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(str::to_string)
}

fn nostr_json_response(name: &str, pubkey: &str) -> Response {
    let mut names = std::collections::BTreeMap::new();
    names.insert(name.to_string(), pubkey.to_string());
    let headers = [(header::CACHE_CONTROL, NOSTR_JSON_CACHE_CONTROL)];
    (headers, Json(NostrJsonResponse { names })).into_response()
}

fn not_found() -> (StatusCode, Json<Value>) {
    (StatusCode::NOT_FOUND, Json(Value::String(String::new())))
}

fn bad_request(message: &str) -> (StatusCode, Json<Value>) {
    (
        StatusCode::BAD_REQUEST,
        Json(Value::String(message.to_string())),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repository::{
        AccountIdentifierKind, AccountProvider, NewAccountIdentifier, NewSparkRegistration,
        ResolvedRecipient,
    };
    use crate::routes::test_support::{
        MockRepository, Request, Router, ServiceExt, StatusCode, Value,
        internal_route_test_state_with_blink_endpoint, json,
        route_test_state_with_country_resolver, start_blink_me_mock_server,
    };
    use nostr::{EventBuilder, Keys, Tag};

    const DOMAIN: &str = "localhost:8080";

    fn proof_event(keys: &Keys, lnaddress: &str, kind: Kind) -> String {
        let event = EventBuilder::new(kind, "")
            .tags([Tag::parse(["lnaddress", lnaddress]).expect("test tag parses")])
            .sign_with_keys(keys)
            .expect("test proof signs");
        event.as_json()
    }

    async fn call(
        app: Router,
        method: &str,
        uri: &str,
        auth: Option<&str>,
        body: Option<Value>,
    ) -> (StatusCode, Value) {
        let mut builder = Request::builder()
            .method(method)
            .uri(uri)
            .header("host", DOMAIN);
        if let Some(auth) = auth {
            builder = builder.header("authorization", auth);
        }
        let request = if let Some(body) = body {
            builder
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body.to_string()))
        } else {
            builder.body(axum::body::Body::empty())
        }
        .expect("test request builds");
        let response = app.oneshot(request).await.expect("route responds");
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body reads");
        let body = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).expect("response body is JSON")
        };
        (status, body)
    }

    #[test]
    fn valid_local_parts_follow_nip05_charset() {
        assert!(valid_nip05_local_part("alice"));
        assert!(valid_nip05_local_part("a-b_c.d01"));
        assert!(valid_nip05_local_part("_"));
        assert!(!valid_nip05_local_part(""));
        assert!(!valid_nip05_local_part("Alice"));
        assert!(!valid_nip05_local_part("alice@blink.sv"));
        assert!(!valid_nip05_local_part("sp ace"));
        assert!(!valid_nip05_local_part(&"x".repeat(65)));
    }

    #[test]
    fn nostr_pubkey_normalization_rejects_noncanonical_forms() {
        let hex = "a1b2".repeat(16);
        assert_eq!(normalize_nostr_pubkey(&hex).unwrap(), hex);
        assert_eq!(normalize_nostr_pubkey(&format!(" {hex} ")).unwrap(), hex);
        assert!(normalize_nostr_pubkey(&hex.to_uppercase()).is_err());
        assert!(normalize_nostr_pubkey("npub1something").is_err());
        assert!(normalize_nostr_pubkey(&hex[..63]).is_err());
        assert!(normalize_nostr_pubkey(&format!("{hex}zz")).is_err());
    }

    #[test]
    fn proof_verifier_accepts_valid_authentication_event() {
        let keys = Keys::generate();
        let proof = proof_event(&keys, "alice@localhost:8080", Kind::Authentication);
        assert!(
            verify_nostr_identity_proof(
                &proof,
                &keys.public_key().to_hex(),
                "alice@localhost:8080"
            )
            .is_ok()
        );
        // The http-auth kind is equally acceptable.
        let proof = proof_event(&keys, "alice@localhost:8080", Kind::HttpAuth);
        assert!(
            verify_nostr_identity_proof(
                &proof,
                &keys.public_key().to_hex(),
                "alice@localhost:8080"
            )
            .is_ok()
        );
    }

    #[test]
    fn proof_verifier_rejects_wrong_kind_tag_or_key() {
        let keys = Keys::generate();
        let other_keys = Keys::generate();
        let pubkey = keys.public_key().to_hex();

        // Wrong kind.
        let proof = proof_event(&keys, "alice@localhost:8080", Kind::TextNote);
        assert!(verify_nostr_identity_proof(&proof, &pubkey, "alice@localhost:8080").is_err());

        // Wrong lnaddress binding.
        let proof = proof_event(&keys, "alice@localhost:8080", Kind::Authentication);
        assert!(verify_nostr_identity_proof(&proof, &pubkey, "mallory@localhost:8080").is_err());

        // Claimed key differs from signing key (impersonation attempt).
        assert!(
            verify_nostr_identity_proof(
                &proof,
                &other_keys.public_key().to_hex(),
                "alice@localhost:8080"
            )
            .is_err()
        );

        // No lnaddress tag at all.
        let event = EventBuilder::new(Kind::Authentication, "")
            .sign_with_keys(&keys)
            .expect("test event signs");
        assert!(
            verify_nostr_identity_proof(&event.as_json(), &pubkey, "alice@localhost:8080").is_err()
        );
    }

    fn nostr_json_app(state: State<MockRepository>) -> Router {
        Router::new()
            .route(
                "/.well-known/nostr.json",
                axum::routing::get(LnurlServer::<MockRepository>::handle_nostr_json),
            )
            .layer(Extension(state))
    }

    #[tokio::test]
    async fn nostr_json_serves_overlay_then_registry() {
        let repo =
            MockRepository::default().with_nostr_identity("acct_1", DOMAIN, &"11".repeat(32));
        let mut state = route_test_state_with_country_resolver(
            repo,
            crate::country::CountryResolver::disabled(),
        )
        .await;
        let mut overlay = std::collections::BTreeMap::new();
        overlay.insert("_".to_string(), "22".repeat(32));
        state.nostr_static_names = std::sync::Arc::new(overlay);
        let app = nostr_json_app(state);

        // Overlay wins and is served verbatim.
        let (status, body) = call(
            app.clone(),
            "GET",
            "/.well-known/nostr.json?name=_",
            None,
            None,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["names"]["_"], "22".repeat(32));
        assert_eq!(
            body["names"].as_object().expect("names is an object").len(),
            1,
            "only the queried name is ever returned"
        );

        // Registry lookup for a registered user.
        let (status, body) = call(
            app.clone(),
            "GET",
            "/.well-known/nostr.json?name=alice",
            None,
            None,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["names"]["alice"], "11".repeat(32));
    }

    #[tokio::test]
    async fn nostr_json_404s_uniformly_for_unknown_invalid_or_missing_names() {
        let repo = MockRepository::default();
        let state = route_test_state_with_country_resolver(
            repo,
            crate::country::CountryResolver::disabled(),
        )
        .await;
        let app = nostr_json_app(state);

        for uri in [
            "/.well-known/nostr.json?name=nobody",
            "/.well-known/nostr.json?name=Alice",
            "/.well-known/nostr.json?name=alice@blink.sv",
            "/.well-known/nostr.json",
        ] {
            let (status, _) = call(app.clone(), "GET", uri, None, None).await;
            assert_eq!(status, StatusCode::NOT_FOUND, "uri: {uri}");
        }
    }

    fn spark_registration(pubkey: &str, username: &str) -> NewSparkRegistration {
        NewSparkRegistration {
            account_id: Some(format!("acct_mock_{pubkey}")),
            pubkey: pubkey.to_string(),
            identifier: NewAccountIdentifier {
                domain: DOMAIN.to_string(),
                identifier: username.to_string(),
                identifier_kind: AccountIdentifierKind::Username,
                description: String::new(),
            },
        }
    }

    #[tokio::test]
    async fn register_nostr_binds_spark_handle_with_dual_proof() {
        let nostr_keys = Keys::generate();
        let nostr_pubkey = nostr_keys.public_key().to_hex();
        // build_auth_payload signs `{username}-{timestamp}`; passing the
        // nostr-prefixed "username" yields exactly the canonical
        // `nostr:{nostr_pubkey}-{timestamp}` message the route validates.
        let timestamp = now_u64();
        let auth =
            spark_client::Client::build_auth_payload(&format!("nostr:{nostr_pubkey}"), timestamp)
                .await
                .expect("test auth payload signs");

        let repo = MockRepository::default().with_spark_mode(&auth.pubkey, None);
        repo.spark_registrations
            .lock()
            .unwrap()
            .push(spark_registration(&auth.pubkey, "alice"));
        let state = route_test_state_with_country_resolver(
            repo.clone(),
            crate::country::CountryResolver::disabled(),
        )
        .await;
        let app = Router::new()
            .route(
                "/lnurlpay/{pubkey}/nostr",
                axum::routing::post(LnurlServer::<MockRepository>::register_nostr),
            )
            .layer(Extension(state));

        let (status, body) = call(
            app.clone(),
            "POST",
            &format!("/lnurlpay/{}/nostr", auth.pubkey),
            None,
            Some(json!({
                "nostr_pubkey": nostr_pubkey,
                "nostr_proof": proof_event(
                    &nostr_keys,
                    "alice@localhost:8080",
                    Kind::Authentication
                ),
                "signature": auth.register_signature,
                "timestamp": timestamp,
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "body: {body}");
        assert_eq!(body["nip05"], "alice@localhost:8080");
        assert_eq!(
            repo.nostr_identity(&format!("acct_mock_{}", auth.pubkey), DOMAIN),
            Some(nostr_pubkey)
        );
    }

    #[tokio::test]
    async fn register_nostr_rejects_bad_signature_and_foreign_proof_key() {
        let nostr_keys = Keys::generate();
        let foreign_keys = Keys::generate();
        let nostr_pubkey = nostr_keys.public_key().to_hex();
        let timestamp = now_u64();
        let auth =
            spark_client::Client::build_auth_payload(&format!("nostr:{nostr_pubkey}"), timestamp)
                .await
                .expect("test auth payload signs");

        let repo = MockRepository::default().with_spark_mode(&auth.pubkey, None);
        repo.spark_registrations
            .lock()
            .unwrap()
            .push(spark_registration(&auth.pubkey, "alice"));
        let state = route_test_state_with_country_resolver(
            repo,
            crate::country::CountryResolver::disabled(),
        )
        .await;
        let app = Router::new()
            .route(
                "/lnurlpay/{pubkey}/nostr",
                axum::routing::post(LnurlServer::<MockRepository>::register_nostr),
            )
            .layer(Extension(state));

        // Invalid spark signature.
        let (status, _) = call(
            app.clone(),
            "POST",
            &format!("/lnurlpay/{}/nostr", auth.pubkey),
            None,
            Some(json!({
                "nostr_pubkey": nostr_pubkey,
                "nostr_proof": proof_event(
                    &nostr_keys,
                    "alice@localhost:8080",
                    Kind::Authentication
                ),
                "signature": "00".repeat(70),
                "timestamp": timestamp,
            })),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        // Proof signed by a different nostr key than the one claimed.
        let (status, _) = call(
            app.clone(),
            "POST",
            &format!("/lnurlpay/{}/nostr", auth.pubkey),
            None,
            Some(json!({
                "nostr_pubkey": nostr_pubkey,
                "nostr_proof": proof_event(
                    &foreign_keys,
                    "alice@localhost:8080",
                    Kind::Authentication
                ),
                "signature": auth.register_signature,
                "timestamp": timestamp,
            })),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    fn blink_recipient(identifier: &str) -> ResolvedRecipient {
        ResolvedRecipient {
            account_id: "acct_blink".to_string(),
            provider: AccountProvider::Blink,
            domain: DOMAIN.to_string(),
            identifier: identifier.to_string(),
            identifier_kind: AccountIdentifierKind::Username,
            description: String::new(),
            spark_pubkey: None,
            blink_account_id: Some("blink-account-1".to_string()),
            btc_wallet_id: None,
            usd_wallet_id: None,
            default_wallet: None,
        }
    }

    #[tokio::test]
    async fn register_nostr_blink_binds_provisioned_username() {
        let endpoint = start_blink_me_mock_server(Some("alice")).await;
        let repo = MockRepository::default().with_resolved_recipient(blink_recipient("alice"));
        let state =
            internal_route_test_state_with_blink_endpoint(repo.clone(), None, &endpoint).await;
        let app = Router::new()
            .route(
                "/nostr/blink",
                axum::routing::post(LnurlServer::<MockRepository>::register_nostr_blink),
            )
            .layer(Extension(state));

        let nostr_keys = Keys::generate();
        let nostr_pubkey = nostr_keys.public_key().to_hex();
        let (status, body) = call(
            app.clone(),
            "POST",
            "/nostr/blink",
            Some("Bearer good-token"),
            Some(json!({
                "nostr_pubkey": nostr_pubkey,
                "nostr_proof": proof_event(
                    &nostr_keys,
                    "alice@localhost:8080",
                    Kind::Authentication
                ),
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "body: {body}");
        assert_eq!(body["nip05"], "alice@localhost:8080");
        assert_eq!(
            repo.nostr_identity("acct_blink", DOMAIN),
            Some(nostr_pubkey)
        );
    }

    #[tokio::test]
    async fn register_nostr_blink_rejects_bad_token_and_unprovisioned_username() {
        let endpoint = start_blink_me_mock_server(Some("alice")).await;

        // Valid token, but the username is not provisioned in the registry.
        let repo = MockRepository::default();
        let state = internal_route_test_state_with_blink_endpoint(repo, None, &endpoint).await;
        let app = Router::new()
            .route(
                "/nostr/blink",
                axum::routing::post(LnurlServer::<MockRepository>::register_nostr_blink),
            )
            .layer(Extension(state));
        let nostr_keys = Keys::generate();
        let (status, _) = call(
            app.clone(),
            "POST",
            "/nostr/blink",
            Some("Bearer good-token"),
            Some(json!({
                "nostr_pubkey": nostr_keys.public_key().to_hex(),
                "nostr_proof": proof_event(
                    &nostr_keys,
                    "alice@localhost:8080",
                    Kind::Authentication
                ),
            })),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);

        // Bad token.
        let repo = MockRepository::default().with_resolved_recipient(blink_recipient("alice"));
        let state = internal_route_test_state_with_blink_endpoint(repo, None, &endpoint).await;
        let app = Router::new()
            .route(
                "/nostr/blink",
                axum::routing::post(LnurlServer::<MockRepository>::register_nostr_blink),
            )
            .layer(Extension(state));
        let (status, _) = call(
            app.clone(),
            "POST",
            "/nostr/blink",
            Some("Bearer wrong-token"),
            Some(json!({
                "nostr_pubkey": nostr_keys.public_key().to_hex(),
                "nostr_proof": proof_event(
                    &nostr_keys,
                    "alice@localhost:8080",
                    Kind::Authentication
                ),
            })),
        )
        .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }
}
