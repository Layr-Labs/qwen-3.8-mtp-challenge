#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: download-r2-object.sh OBJECT_PATH OUTPUT_PATH" >&2
  exit 2
fi

object_path="${1#/}"
output_path="$2"

# Defense in depth: the object key is a trusted workflow constant today, but
# validate it before it is signed into the SigV4 canonical request and the
# request URL so a future caller cannot smuggle path traversal (`..`/`.`
# segments), an absolute path, or control characters (a newline/CR would
# corrupt the canonical request or the HTTP request line) into the R2 request.
if [[ -z "${object_path}" ]]; then
  echo "download-r2-object: object path must not be empty" >&2
  exit 2
fi
if [[ "${object_path}" == /* ]]; then
  echo "download-r2-object: object path must not be absolute: ${object_path}" >&2
  exit 2
fi
case "/${object_path}/" in
  *"/../"*|*"/./"*)
    echo "download-r2-object: object path must not contain . or .. segments: ${object_path}" >&2
    exit 2
    ;;
esac
if [[ "${object_path}" =~ [[:cntrl:]] ]]; then
  echo "download-r2-object: object path must not contain control characters" >&2
  exit 2
fi
# Nothing here percent-encodes. The same raw key bytes are signed into the
# SigV4 canonical request AND handed to curl, so the script is only correct
# while identity encoding is the right encoding. For the keys the workflows
# actually fetch ([A-Za-z0-9._/-]) it is. Outside that set it is not, and each
# way it breaks is quiet:
#
#   ' '        curl exit 3, "URL rejected: Malformed input"
#   '#'        curl truncates the URL at the fragment -> wrong key, or 403
#   '?'        the tail becomes a query string, but the canonical query
#              string is signed as EMPTY -> 403 SignatureDoesNotMatch
#   non-ASCII  curl percent-encodes it on the wire AFTER it was signed raw
#              -> 403 SignatureDoesNotMatch against a healthy bucket
#
# The fix is NOT to add an RFC 3986 encoder: the aws-CLI branch below encodes
# by botocore's rules, and a second encoder here that disagreed with it would
# be a fresh divergence between the two branches for the same key. Enforce the
# assumption instead, so an unsupported key is a clear refusal before any
# request rather than a 403 that reads like a credentials fault.
object_path_pattern='^[A-Za-z0-9._/-]+$'
if [[ ! "${object_path}" =~ ${object_path_pattern} ]]; then
  echo "download-r2-object: object path must match [A-Za-z0-9._/-]+ because the key bytes are signed and sent verbatim with no percent-encoding: ${object_path}" >&2
  exit 2
fi

: "${R2_ACCESS_KEY_ID:?R2_ACCESS_KEY_ID is required}"
: "${R2_BUCKET_ENDPOINT:?R2_BUCKET_ENDPOINT is required}"
: "${R2_SECRET_ACCESS_KEY:?R2_SECRET_ACCESS_KEY is required}"

# Describe a value without revealing it: byte length plus the hex of the
# first and last byte. Enough to spot an invisible poison byte (0x0a
# newline, 0x0d CR, 0xa0 from a pasted NBSP) in a secret from the run log.
value_fingerprint() {
  local value="$1"
  local length
  length="$(LC_ALL=C printf '%s' "${value}" | wc -c | tr -d '[:space:]')"
  if [[ "${length}" == "0" ]]; then
    printf 'length=0'
    return 0
  fi
  local first_hex last_hex
  first_hex="$(LC_ALL=C printf '%s' "${value}" | head -c 1 | xxd -p | tr -d '\n')"
  last_hex="$(LC_ALL=C printf '%s' "${value}" | tail -c 1 | xxd -p | tr -d '\n')"
  printf 'length=%s first=0x%s last=0x%s' "${length}" "${first_hex}" "${last_hex}"
}

# Remove bytes that can never appear in a valid https endpoint: every ASCII
# whitespace/control byte (the trailing newline `echo | gh secret set`
# injects, CRs from Windows clipboards, tabs) plus the UTF-8 non-breaking
# space bytes (0xC2 0xA0) that rich-text editors paste. A valid endpoint is
# pure ASCII, so byte-wise deletion cannot corrupt a legitimate value; a
# poisoned-but-otherwise-correct secret is rescued instead of handing curl a
# malformed URL (curl exit 3, "URL rejected").
strip_invisible_bytes() {
  LC_ALL=C printf '%s' "$1" | LC_ALL=C tr -d '[:space:][:cntrl:]\302\240'
}

raw_bucket_endpoint="${R2_BUCKET_ENDPOINT}"
endpoint="$(strip_invisible_bytes "${raw_bucket_endpoint}")"
if [[ "${endpoint}" != "${raw_bucket_endpoint}" ]]; then
  echo "download-r2-object: stripped invisible whitespace/control bytes from R2_BUCKET_ENDPOINT ($(value_fingerprint "${raw_bucket_endpoint}"))" >&2
fi
endpoint="${endpoint%/}"
endpoint_pattern='^https://[a-z0-9.-]+(/[A-Za-z0-9._-]+)*$'
if [[ ! "${endpoint}" =~ ${endpoint_pattern} ]]; then
  echo "download-r2-object: R2_BUCKET_ENDPOINT is not a valid https R2 endpoint after sanitization ($(value_fingerprint "${raw_bucket_endpoint}")); expected https://<host>[/<bucket>[/<prefix>...]]; refusing to print the value" >&2
  exit 1
fi

# The same paste class poisons the credentials: an invisible byte inside
# R2_ACCESS_KEY_ID lands mid-string in the Authorization header (curl drops
# the malformed header and the request arrives unsigned -> R2 answers 400
# InvalidArgument: Authorization), and one inside R2_SECRET_ACCESS_KEY feeds
# the HMAC a wrong key (403 SignatureDoesNotMatch). Neither byte class is
# ever legitimate in an R2 credential, so strip and note it. The secret's
# notice reports length only - never any byte of the secret.
raw_access_key_id="${R2_ACCESS_KEY_ID}"
R2_ACCESS_KEY_ID="$(strip_invisible_bytes "${raw_access_key_id}")"
if [[ "${R2_ACCESS_KEY_ID}" != "${raw_access_key_id}" ]]; then
  echo "download-r2-object: stripped invisible whitespace/control bytes from R2_ACCESS_KEY_ID ($(value_fingerprint "${raw_access_key_id}"))" >&2
fi
access_key_pattern='^[A-Za-z0-9]+$'
if [[ ! "${R2_ACCESS_KEY_ID}" =~ ${access_key_pattern} ]]; then
  echo "download-r2-object: R2_ACCESS_KEY_ID is not a plausible access key id after sanitization ($(value_fingerprint "${raw_access_key_id}")); refusing to print the value" >&2
  exit 1
fi

raw_secret_access_key="${R2_SECRET_ACCESS_KEY}"
R2_SECRET_ACCESS_KEY="$(strip_invisible_bytes "${raw_secret_access_key}")"
if [[ "${R2_SECRET_ACCESS_KEY}" != "${raw_secret_access_key}" ]]; then
  echo "download-r2-object: stripped invisible whitespace/control bytes from R2_SECRET_ACCESS_KEY (length=$(LC_ALL=C printf '%s' "${raw_secret_access_key}" | wc -c | tr -d '[:space:]'); byte values withheld)" >&2
fi
if [[ -z "${R2_SECRET_ACCESS_KEY}" ]]; then
  echo "download-r2-object: R2_SECRET_ACCESS_KEY is empty after sanitization" >&2
  exit 1
fi

endpoint_rest="${endpoint#https://}"
host="${endpoint_rest%%/*}"
base_path="${endpoint_rest#"${host}"}"
request_path="${base_path}/${object_path}"
url="https://${host}${request_path}"

region="${R2_REGION:-auto}"
service="s3"
payload_hash="$(printf '' | shasum -a 256 | awk '{print $1}')"
signed_headers="host;x-amz-content-sha256;x-amz-date"

hmac_hex() {
  local key_opt="$1"
  local message="$2"
  printf '%s' "${message}" \
    | openssl dgst -sha256 -mac HMAC -macopt "${key_opt}" -binary \
    | xxd -p -c 256
}

# Produce a COMPLETE, self-consistent SigV4 signature for one network attempt:
# amz_date, the canonical request built from it, and the Authorization header
# that covers both. Every retry calls this again.
#
# It used to run once, before curl, and curl owned the retry (`--retry 5
# --retry-all-errors`). curl replays the argv it was handed, and the
# Authorization / x-amz-date headers are already fixed in that argv -- measured
# against real R2, all six attempts carried the SAME x-amz-date. A SigV4
# signature is only accepted inside R2's ~15 minute clock-skew window, so a run
# that spends long enough in retries turns a transient 503 into a
# permanent-looking 403 RequestTimeTooSkewed. curl cannot fix that: only the
# caller can re-sign. Nothing that varies per attempt may be hoisted out of
# this function.
sign_request() {
  amz_date="$(date -u +%Y%m%dT%H%M%SZ)"
  date_stamp="${amz_date:0:8}"
  # SigV4 CanonicalRequest is
  #   METHOD \n URI \n QUERY \n CanonicalHeaders \n SignedHeaders \n PayloadHash
  # and CanonicalHeaders is itself "name:value\n" per header -- so a BLANK LINE
  # separates the last header from SignedHeaders. This used to spell that
  # terminating newline inside canonical_headers' own printf, where command
  # substitution ATE it: `$(...)` strips every trailing newline, so the canonical
  # request went on the wire one line short, hashed differently from what R2
  # computed for the same request, and R2 answered HTTP 403 SignatureDoesNotMatch
  # with a well-formed signature. Never reached on the serial box because its
  # runner PATH has the aws CLI, which takes download_with_aws_cli() instead;
  # M5-C's runner PATH is /usr/bin:/bin:/usr/sbin:/sbin, so it is the first box to
  # execute this signer at all (2026-07-30).
  #
  # Both newlines are therefore spelled in the canonical_request format string
  # ("...%s\n\n%s..."), where nothing can strip them: one terminates the last
  # header line, one is the blank separator.
  canonical_headers="$(printf 'host:%s\nx-amz-content-sha256:%s\nx-amz-date:%s' "${host}" "${payload_hash}" "${amz_date}")"
  canonical_request="$(printf 'GET\n%s\n\n%s\n\n%s\n%s' "${request_path}" "${canonical_headers}" "${signed_headers}" "${payload_hash}")"
  credential_scope="${date_stamp}/${region}/${service}/aws4_request"
  canonical_request_hash="$(printf '%s' "${canonical_request}" | shasum -a 256 | awk '{print $1}')"
  string_to_sign="$(printf 'AWS4-HMAC-SHA256\n%s\n%s\n%s' "${amz_date}" "${credential_scope}" "${canonical_request_hash}")"
  k_date="$(hmac_hex "key:AWS4${R2_SECRET_ACCESS_KEY}" "${date_stamp}")"
  k_region="$(hmac_hex "hexkey:${k_date}" "${region}")"
  k_service="$(hmac_hex "hexkey:${k_region}" "${service}")"
  k_signing="$(hmac_hex "hexkey:${k_service}" "aws4_request")"
  # Signed through hmac_hex, exactly like the four key-derivation steps above.
  # It used to run its own openssl pipeline WITHOUT -binary and parse the text
  # form with `awk '{print $2}'`, which silently depends on the openssl
  # implementation's output format:
  #
  #   OpenSSL 3.x   "SHA2-256(stdin)= <hex>"  -> $2 is the hex
  #   LibreSSL 3.3  "<hex>"                   -> $2 is EMPTY, the hex is $1
  #
  # macOS ships LibreSSL as /usr/bin/openssl, so on a box without Homebrew
  # OpenSSL ahead of it the signature came out EMPTY and R2 answered
  # "InvalidArgument: Signature element value should not be blank" -- a 400 that
  # looks like a credentials problem and is not one. Diagnosed on M5-C
  # (LibreSSL 3.3.6) 2026-07-30.
  #
  # That diagnosis originally added "the serial box worked only because OpenSSL 3
  # was first on its PATH." That is FALSE, and believing it sends the next
  # debugger to audit PATH ordering on a box that never runs this code. The
  # serial box has the aws CLI on its runner PATH, so download_with_aws_cli()
  # below returns 0 and the signer is never reached: every successful hidden-
  # golden fetch in either repo announced "using AWS CLI S3 path-style download",
  # never "using signed HTTPS download". Whichever openssl it ships is
  # irrelevant. R2_FORCE_SIGNED=1 now gives this path an execution environment
  # that is not a production run; use it, and keep using it.
  #
  # -binary sidesteps the text format entirely, so there is no field to index.
  signature="$(hmac_hex "hexkey:${k_signing}" "${string_to_sign}")"
  authorization="AWS4-HMAC-SHA256 Credential=${R2_ACCESS_KEY_ID}/${credential_scope}, SignedHeaders=${signed_headers}, Signature=${signature}"
}

# Attempt budget. Six attempts is what `--retry 5` gave, kept so the change is
# to WHO retries, not to how many times. The delay is overridable so tests can
# exercise the loop's shape without its wall clock.
max_attempts=6
retry_delay_seconds="${R2_RETRY_DELAY_SECONDS:-2}"
retry_delay_pattern='^[0-9]+$'
if [[ ! "${retry_delay_seconds}" =~ ${retry_delay_pattern} ]]; then
  echo "download-r2-object: R2_RETRY_DELAY_SECONDS must be a non-negative integer" >&2
  exit 2
fi

tmp_path="${output_path}.tmp"

umask 077
mkdir -p "$(dirname "${output_path}")"
trap 'rm -f "${tmp_path}"' EXIT

download_with_aws_cli() {
  local trimmed_path="${base_path#/}"
  local bucket
  local key_prefix
  local key

  if [[ -z "${trimmed_path}" || ! -x "$(command -v aws 2>/dev/null)" ]]; then
    return 1
  fi

  bucket="${trimmed_path%%/*}"
  key_prefix="${trimmed_path#"${bucket}"}"
  key_prefix="${key_prefix#/}"
  key="${object_path}"
  if [[ -n "${key_prefix}" ]]; then
    key="${key_prefix}/${object_path}"
  fi

  echo "download-r2-object: using AWS CLI S3 path-style download"
  AWS_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID}" \
    AWS_SECRET_ACCESS_KEY="${R2_SECRET_ACCESS_KEY}" \
    AWS_DEFAULT_REGION="${region}" \
    AWS_EC2_METADATA_DISABLED=true \
    aws \
      --endpoint-url "https://${host}" \
      s3 cp \
      "s3://${bucket}/${key}" \
      "${tmp_path}" \
      --only-show-errors \
      --no-progress
}

# Which implementation actually performs a transfer is NOT fixed:
#
#   * `aws` absent (or the endpoint has no bucket path) -> signed path;
#   * `aws` present and succeeds                        -> aws path;
#   * `aws` present but FAILS (missing plugin, bad config, transient error)
#     -> download_with_aws_cli returns non-zero and this script falls
#     through to the signed path SILENTLY.
#
# So a box can take either branch on different runs of the same workflow, and
# the only record of which one ran is the banner each branch prints. That is
# also how the signer stayed unexecuted long enough to ship two signing bugs:
# it is PREFERRED against, never selected. R2_FORCE_SIGNED=1 skips the
# fallback so the signed path can be exercised deliberately -- by the tests in
# Tests/MLXFastTests/R2RequestExecutionTests.swift, and by hand on any box.
if [[ "${R2_FORCE_SIGNED:-0}" != "1" ]] && download_with_aws_cli; then
  chmod 600 "${tmp_path}"
  mv "${tmp_path}" "${output_path}"
  trap - EXIT
  echo "download-r2-object: wrote ${output_path}"
  exit 0
fi

echo "download-r2-object: using signed HTTPS download"
curl_rc=0
http_status=""
attempt=1
while :; do
  # Truncate before every attempt. curl truncates --output only once it OPENS
  # that file, which it never does when the attempt dies at connect time or
  # times out -- so without this a failed attempt's <Error> document survives
  # under, or ahead of, the bytes the next attempt writes. The caller verifies
  # a pinned sha256 and would report a golden mismatch for a transport
  # artefact.
  : > "${tmp_path}"
  # Fresh amz_date, canonical request and signature for THIS attempt.
  sign_request
  curl_rc=0
  # --connect-timeout / --max-time are what make an attempt terminate at all.
  # There was no bound: a peer that accepts the connection and then sends
  # nothing triggers neither a retry nor a timeout, so the transfer "succeeds"
  # with an empty body, or the job hangs to its 30-minute limit with no
  # diagnosis. --retry/--retry-all-errors/--retry-delay are deliberately gone:
  # a curl-level retry replays fixed argv and therefore cannot re-sign.
  # --location is deliberately gone too: on a cross-host redirect curl DROPS
  # the custom Authorization header but FORWARDS x-amz-date and
  # x-amz-content-sha256, so the follow-up arrives unsigned with signing
  # headers attached, and on a same-host redirect it re-sends a signature
  # bound to the OLD path. R2 path-style issues no legitimate redirects, so
  # following one can only turn a loud failure into a confusing one.
  http_status="$(
    curl \
      --fail-with-body \
      --silent \
      --show-error \
      --connect-timeout 30 \
      --max-time 600 \
      --write-out '%{http_code}' \
      -H "Authorization: ${authorization}" \
      -H "x-amz-content-sha256: ${payload_hash}" \
      -H "x-amz-date: ${amz_date}" \
      --output "${tmp_path}" \
      "${url}"
  )" || curl_rc=$?
  if [[ "${curl_rc}" -eq 0 ]]; then
    break
  fi
  if [[ "${attempt}" -ge "${max_attempts}" ]]; then
    break
  fi
  echo "download-r2-object: attempt ${attempt}/${max_attempts} failed (curl exit ${curl_rc}, HTTP ${http_status:-none}); re-signing and retrying" >&2
  attempt=$((attempt + 1))
  if [[ "${retry_delay_seconds}" -gt 0 ]]; then
    sleep "${retry_delay_seconds}"
  fi
done
if [[ "${curl_rc}" -ne 0 ]]; then
  # --fail-with-body keeps --fail's retry/exit semantics but preserves the
  # response body for diagnostics. The body is only PROVABLY an R2/S3 error
  # document (Code/Message XML: SignatureDoesNotMatch vs InvalidAccessKeyId
  # vs NoSuchBucket vs NoSuchKey) when the final attempt's HTTP status is
  # >= 400 (curl exit 22 under --fail-with-body). Any other failure -- above
  # all a 200 whose transfer died mid-body (curl exit 18/56 through every
  # retry) -- leaves a prefix of the OBJECT ITSELF in the temp file, and this
  # script downloads raw hidden material (correctness golden, GPQA
  # reference) whose bytes must never reach the run log. For those failures
  # report status, exit code, and byte count only.
  http_status_pattern='^[0-9]{3}$'
  if [[ ! "${http_status}" =~ ${http_status_pattern} ]]; then
    http_status="none"
  fi
  received_bytes=0
  if [[ -f "${tmp_path}" ]]; then
    received_bytes="$(wc -c < "${tmp_path}" | tr -d '[:space:]')"
  fi
  if [[ "${http_status}" != "none" ]] && (( 10#${http_status} >= 400 )); then
    error_body="$(LC_ALL=C tr -d '[:cntrl:]' < "${tmp_path}" 2>/dev/null | head -c 400 || true)"
    echo "download-r2-object: R2 request failed (curl exit ${curl_rc}, HTTP ${http_status}); server error body: ${error_body:-<none captured>}" >&2
  else
    echo "download-r2-object: R2 transfer failed (curl exit ${curl_rc}, HTTP ${http_status}, ${received_bytes} body byte(s) discarded); body withheld -- only an HTTP >= 400 response is provably a server error document rather than truncated object content" >&2
  fi
  exit "${curl_rc}"
fi

chmod 600 "${tmp_path}"
mv "${tmp_path}" "${output_path}"
trap - EXIT
echo "download-r2-object: wrote ${output_path}"
