"""
Daily Review Response Agent for Luca Money
============================================
Reads reviews from App Store Connect (iOS) and Android (cache/data.json),
generates personalized responses using Gemini, and emails a daily report.

Usage:
    python agent.py

After reviewing the email, use approve.py to post approved responses.
"""

import os
import json
import time
import smtplib
import datetime
import jwt
import requests
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from google import genai
from google.genai import types

# ─── CONFIG (loaded from environment) ────────────────────────────────────────

IOS_ISSUER_ID  = os.environ["IOS_ISSUER_ID"]
IOS_KEY_ID     = os.environ["IOS_KEY_ID"]
IOS_APP_ID     = os.environ["IOS_APP_ID"]
IOS_P8_PATH    = os.path.expanduser(os.environ["IOS_P8_PATH"])

ANDROID_PACKAGE      = os.environ.get("ANDROID_PACKAGE", "com.undr.luca")
ANDROID_SERVICE_ACCT = os.path.expanduser(os.environ["ANDROID_SERVICE_ACCT"])

GMAIL_USER     = os.environ["GMAIL_USER"]
GMAIL_APP_PASS = os.environ["GMAIL_APP_PASS"]
REPORT_EMAIL   = os.environ["REPORT_EMAIL"]

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

REPORTS_DIR   = Path(__file__).parent / "reports"
FEEDBACK_FILE = Path(__file__).parent / "feedback.json"
REPORTS_DIR.mkdir(exist_ok=True)

RESPONSE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "platform": {"type": "string", "enum": ["ios", "android"]},
            "response": {"type": "string"},
        },
        "required": ["id", "platform", "response"],
    },
}


def load_feedback_examples() -> str:
    """Load past edited responses to teach the agent Tomas's style."""
    if not FEEDBACK_FILE.exists():
        return ""
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        examples = json.load(f)
    if not examples:
        return ""

    lines = ["\n\nIMPORTANT — Here are real examples of responses that were corrected by the team. Learn from these to match our exact tone and style:\n"]
    for ex in examples[-10:]:  # use last 10 edits max
        lines.append(f"Review: \"{ex['review'][:120]}\"")
        lines.append(f"Bad response:     {ex['original_response']}")
        lines.append(f"Correct response: {ex['corrected_response']}")
        lines.append("")
    lines.append("Always write responses that match the style of the 'Correct response' examples above.")
    return "\n".join(lines)

# ─── IOS APP STORE CONNECT ───────────────────────────────────────────────────

def _make_ios_token():
    """Generate a signed JWT for App Store Connect API."""
    with open(IOS_P8_PATH, "r") as f:
        private_key = f.read()

    now = int(time.time())
    payload = {
        "iss": IOS_ISSUER_ID,
        "iat": now,
        "exp": now + 1200,   # 20-minute expiry
        "aud": "appstoreconnect-v1",
    }
    token = jwt.encode(
        payload,
        private_key,
        algorithm="ES256",
        headers={"kid": IOS_KEY_ID},
    )
    return token


def _ios_review_has_response(review_id: str, headers: dict) -> bool:
    """Returns True if this iOS review already has a developer response."""
    url = f"https://api.appstoreconnect.apple.com/v1/customerReviews/{review_id}/response"
    resp = requests.get(url, headers=headers)
    if resp.status_code == 404:
        return False  # No response exists
    if not resp.ok:
        return False  # On any other error, assume no response
    data = resp.json()
    return data.get("data") is not None  # data=null means no response


def fetch_ios_reviews():
    """Fetch only unanswered customer reviews from App Store Connect.
    Fetches all recent reviews, then calls each review's /response endpoint
    to check if it already has a developer response (the only reliable Apple API method).
    """
    print("📱 Fetching iOS reviews from App Store Connect...")
    token = _make_ios_token()
    headers = {"Authorization": f"Bearer {token}"}
    url = f"https://api.appstoreconnect.apple.com/v1/apps/{IOS_APP_ID}/customerReviews"
    params = {
        "sort": "-createdDate",
        "limit": 50,  # Most recent 50 reviews is enough for daily runs
        "fields[customerReviews]": "rating,title,body,reviewerNickname,createdDate,territory",
    }

    candidates = []
    while url:
        resp = requests.get(url, headers=headers, params=params)
        if not resp.ok:
            print(f"   ❌ Apple API error {resp.status_code}: {resp.text[:500]}")
            resp.raise_for_status()
        data = resp.json()

        for item in data.get("data", []):
            attrs = item["attributes"]
            candidates.append({
                "platform": "ios",
                "id": item["id"],
                "author": attrs.get("reviewerNickname", "Anonymous"),
                "rating": attrs.get("rating", 0),
                "title": attrs.get("title", ""),
                "body": attrs.get("body", ""),
                "date": attrs.get("createdDate", ""),
                "territory": attrs.get("territory", ""),
            })

        # Only fetch first page (most recent 50) — no need to paginate for daily run
        break

    # Check each review individually for an existing response
    print(f"   Checking {len(candidates)} reviews for existing responses...")
    reviews = []
    for i, review in enumerate(candidates):
        has_resp = _ios_review_has_response(review["id"], headers)
        if not has_resp:
            reviews.append(review)
        if (i + 1) % 10 == 0:
            print(f"   ... checked {i + 1}/{len(candidates)}")

    print(f"   ✓ Found {len(reviews)} unanswered iOS reviews")
    return reviews


def fetch_android_reviews():
    """Fetch only unanswered Android reviews directly from Google Play API."""
    print("🤖 Fetching Android reviews from Google Play...")
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError:
        print("   ⚠️  google-api-python-client not installed. Run: pip3 install google-api-python-client google-auth")
        return []

    creds = service_account.Credentials.from_service_account_file(
        ANDROID_SERVICE_ACCT,
        scopes=["https://www.googleapis.com/auth/androidpublisher"],
    )
    service = build("androidpublisher", "v3", credentials=creds)
    result = service.reviews().list(packageName=ANDROID_PACKAGE, maxResults=100).execute()

    reviews = []
    for item in result.get("reviews", []):
        comments = item.get("comments", [])

        # Skip if already has a developer response
        has_response = any("developerComment" in c for c in comments)
        if has_response:
            continue

        # Get the user's comment
        user_comment = next((c["userComment"] for c in comments if "userComment" in c), {})
        body = user_comment.get("text", "").strip()

        # Skip stars-only reviews with no text
        if not body:
            continue

        reviews.append({
            "platform": "android",
            "id": item.get("reviewId", ""),
            "author": item.get("authorName", "Anonymous"),
            "rating": user_comment.get("starRating", 0),
            "title": "",
            "body": body,
            "date": str(user_comment.get("lastModified", {}).get("seconds", "")),
            "territory": user_comment.get("reviewerLanguage", ""),
        })

    print(f"   ✓ Found {len(reviews)} unanswered Android reviews")
    return reviews


# ─── RESPONSE GENERATION (Gemini) ────────────────────────────────────────────

def generate_responses_with_gemini(reviews: list) -> list:
    """Use Gemini to generate personalized responses for each review."""
    if not reviews:
        return []

    print("\n🤖 Starting Gemini to generate responses...")
    client = genai.Client(api_key=GEMINI_API_KEY)
    feedback_examples = load_feedback_examples()

    system_instruction = """You are the community manager for Luca Money, a fintech savings app in Latin America.
Your job is to write warm, personalized, professional responses to App Store and Google Play reviews.

Rules:
- ALWAYS respond in the SAME LANGUAGE as the review (Spanish or English)
- Keep responses SHORT: 2-4 sentences max
- Be genuine and human, never robotic or generic
- For negative reviews (1-3 stars): acknowledge the issue, apologize sincerely, and invite them to reach out to mariano@lucamoney.com
- For positive reviews (4-5 stars): thank them genuinely, mention a specific detail from their review
- Never copy-paste the same response for multiple reviews
- Do NOT use the "—" symbol anywhere in the response
- Sign off with: "Equipo Luca Money" (if Spanish) or "Luca Money Team" (if English)
""" + feedback_examples

    reviews_text = json.dumps(reviews, ensure_ascii=False, indent=2)
    task = f"""Here are today's app reviews that need personalized responses.

Return your answer as a JSON array where each object has:
- "id": the review id (copy exactly from input)
- "platform": "ios" or "android"
- "response": your written response

Reviews:
{reviews_text}

Return ONLY the JSON array, no extra text."""

    try:
        print("   Gemini is working", end="", flush=True)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=task,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.4,
                max_output_tokens=8192,
                response_mime_type="application/json",
                response_json_schema=RESPONSE_SCHEMA,
            ),
        )
        print(" ✓")
        responses = response.parsed if response.parsed is not None else json.loads(response.text)
    except Exception as exc:
        print(f" ✗\n   ⚠️  Gemini generation failed: {exc}")
        responses = []

    return responses


# ─── MERGE REVIEWS + RESPONSES ───────────────────────────────────────────────

def build_report(reviews: list, responses: list) -> dict:
    """Combine reviews and their generated responses into a report."""
    response_map = {r["id"]: r["response"] for r in responses}

    items = []
    for review in reviews:
        items.append({
            **review,
            "proposed_response": response_map.get(review["id"], ""),
            "approved": False,
        })

    # Sort: bad reviews (1-3 stars) first, then good ones (4-5 stars)
    items.sort(key=lambda r: (0 if r.get("rating", 5) <= 3 else 1, r.get("rating", 5)))

    return {
        "date": datetime.date.today().isoformat(),
        "generated_at": datetime.datetime.now().isoformat(),
        "total_reviews": len(items),
        "ios_count": sum(1 for r in items if r["platform"] == "ios"),
        "android_count": sum(1 for r in items if r["platform"] == "android"),
        "items": items,
    }


def save_report(report: dict) -> Path:
    """Save report to JSON file."""
    filename = REPORTS_DIR / f"report_{report['date']}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Report saved to {filename}")
    return filename


# ─── EMAIL ───────────────────────────────────────────────────────────────────

STAR = "⭐"

def _stars(n):
    return STAR * int(n) + "☆" * (5 - int(n))

def _platform_badge(platform):
    return "🍎 iOS" if platform == "ios" else "🤖 Android"

def _rating_color(rating):
    if rating >= 4:
        return "#22c55e"  # green
    elif rating == 3:
        return "#f59e0b"  # amber
    else:
        return "#ef4444"  # red


def build_email_html(report: dict) -> str:
    today = report["date"]
    total = report["total_reviews"]
    ios_n = report["ios_count"]
    android_n = report["android_count"]

    rows = ""
    for item in report["items"]:
        rating = item.get("rating", 0)
        color = _rating_color(rating)
        rows += f"""
        <tr>
          <td style="padding:16px;border-bottom:1px solid #f0f0f0;vertical-align:top;width:50%;">
            <div style="margin-bottom:6px;">
              <span style="background:{color};color:white;border-radius:4px;padding:2px 8px;font-size:12px;font-weight:bold;">{_stars(rating)}</span>
              <span style="margin-left:8px;font-size:12px;color:#888;">{_platform_badge(item['platform'])}</span>
              <span style="margin-left:8px;font-size:12px;color:#888;">{item.get('author','?')}</span>
            </div>
            {"<div style='font-weight:bold;margin-bottom:4px;'>" + item.get('title','') + "</div>" if item.get('title') else ""}
            <div style="color:#444;font-size:14px;">{item.get('body','(no text)')}</div>
          </td>
          <td style="padding:16px;border-bottom:1px solid #f0f0f0;vertical-align:top;background:#fafafa;">
            <div style="font-size:13px;color:#333;font-style:italic;">"{item.get('proposed_response','—')}"</div>
          </td>
        </tr>"""

    return f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family:Arial,sans-serif;max-width:900px;margin:0 auto;color:#222;">
  <div style="background:#1a1a2e;color:white;padding:24px;border-radius:8px 8px 0 0;">
    <h1 style="margin:0;font-size:22px;">📱 Luca Money — Daily Review Report</h1>
    <p style="margin:4px 0 0;opacity:0.7;">{today} &nbsp;·&nbsp; {total} reviews ({ios_n} iOS, {android_n} Android)</p>
  </div>

  <div style="padding:16px 0;">
    <p style="color:#555;">Below are today's reviews and the proposed responses.
    Open <code>approve.py</code> to approve and post them to the stores.</p>

    <table style="width:100%;border-collapse:collapse;border:1px solid #e5e5e5;border-radius:8px;overflow:hidden;">
      <thead>
        <tr style="background:#f5f5f5;">
          <th style="padding:12px 16px;text-align:left;font-size:13px;color:#666;">Review</th>
          <th style="padding:12px 16px;text-align:left;font-size:13px;color:#666;">Proposed Response</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>

  <div style="background:#f9f9f9;border:1px solid #e5e5e5;border-radius:8px;padding:16px;margin-top:16px;">
    <strong>Next step:</strong> Run <code>python approve.py</code> in your terminal to approve responses interactively,
    or edit the report JSON directly and set <code>"approved": true</code> for the ones you want to post.
  </div>

  <p style="color:#aaa;font-size:12px;margin-top:24px;">Generated by your Luca Money Review Agent · {report['generated_at']}</p>
</body>
</html>"""


def send_email(report: dict, report_path: Path):
    """Send the daily report email via Gmail SMTP."""
    print("\n📧 Sending email report...")
    today = report["date"]
    total = report["total_reviews"]

    recipients = [REPORT_EMAIL, "mariano.maisterrena@yummysuperapp.com"]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[Luca Reviews] {total} new reviews — {today}"
    msg["From"] = GMAIL_USER
    msg["To"] = ", ".join(recipients)

    html = build_email_html(report)
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_USER, GMAIL_APP_PASS)
        server.sendmail(GMAIL_USER, recipients, msg.as_string())

    print(f"   ✓ Email sent to {', '.join(recipients)}")


def _send_no_reviews_email():
    """Send a simple email when there are no new reviews to respond to."""
    today = datetime.date.today().isoformat()
    recipients = [REPORT_EMAIL, "mariano.maisterrena@yummysuperapp.com"]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[Luca Reviews] NO REVIEWS — {today}"
    msg["From"] = GMAIL_USER
    msg["To"] = ", ".join(recipients)

    html = f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;color:#222;">
  <div style="background:#1a1a2e;color:white;padding:24px;border-radius:8px 8px 0 0;">
    <h1 style="margin:0;font-size:22px;">📱 Luca Money — Daily Review Report</h1>
    <p style="margin:4px 0 0;opacity:0.7;">{today}</p>
  </div>
  <div style="padding:40px 24px;text-align:center;">
    <div style="font-size:48px;">🎉</div>
    <h2 style="color:#22c55e;margin:16px 0 8px;">NO REVIEWS</h2>
    <p style="color:#666;font-size:16px;">All reviews have been responded to. Nothing to do today!</p>
  </div>
  <p style="color:#aaa;font-size:12px;text-align:center;">Generated by your Luca Money Review Agent</p>
</body>
</html>"""

    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_USER, GMAIL_APP_PASS)
        server.sendmail(GMAIL_USER, recipients, msg.as_string())

    print(f"   ✓ 'No reviews' email sent to {', '.join(recipients)}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Luca Money Review Agent — Daily Run")
    print(f"  Date: {datetime.date.today()}")
    print("=" * 60)

    # 1. Fetch reviews
    ios_reviews     = fetch_ios_reviews()
    android_reviews = fetch_android_reviews()
    all_reviews     = ios_reviews + android_reviews

    if not all_reviews:
        print("\n✅ No new reviews found today. Sending 'no reviews' email...")
        _send_no_reviews_email()
        return

    # 2. Generate responses with Gemini
    responses = generate_responses_with_gemini(all_reviews)

    # 3. Build + save report
    report = build_report(all_reviews, responses)
    report_path = save_report(report)

    # 4. Send email
    send_email(report, report_path)

    print("\n✅ Done! Check your email and then run: python approve.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
