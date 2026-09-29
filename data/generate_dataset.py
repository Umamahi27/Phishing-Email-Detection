"""Generate a labelled sample dataset of phishing and legitimate emails.

The emails are built from templates with randomised names, companies, URLs and
phrasing, plus some deliberately "hard" examples (legitimate security notices,
phishing written in a calm tone) so the classifier cannot rely on a single cue.

Usage:
    python data/generate_dataset.py --n 2000 --out data/emails.csv
"""

import argparse
import csv
import random
from pathlib import Path

NAMES = ["Alex", "Jordan", "Sam", "Priya", "Chen", "Maria", "Ahmed", "Olivia",
         "Ravi", "Emma", "Lucas", "Aisha", "Noah", "Mei", "Carlos", "Fatima"]
BRANDS = ["PayPal", "Amazon", "Netflix", "Microsoft", "Apple", "Chase Bank",
          "Wells Fargo", "DHL", "FedEx", "Google", "Dropbox", "LinkedIn"]
LEGIT_DOMAINS = ["paypal.com", "amazon.com", "netflix.com", "microsoft.com",
                 "apple.com", "chase.com", "github.com", "google.com",
                 "dropbox.com", "linkedin.com", "company.com", "university.edu"]
SUSPICIOUS_TLDS = ["xyz", "top", "info", "ru", "tk", "click", "online", "biz"]
PHISH_WORDS = ["secure", "verify", "login", "account", "update", "support",
               "billing", "confirm", "alert", "service", "auth", "signin"]


def phishing_url(brand: str) -> str:
    b = brand.lower().replace(" ", "")
    style = random.randint(0, 5)
    word = random.choice(PHISH_WORDS)
    tld = random.choice(SUSPICIOUS_TLDS)
    if style == 0:  # raw IP address
        ip = ".".join(str(random.randint(1, 254)) for _ in range(4))
        return f"http://{ip}/{b}/{word}.php"
    if style == 1:  # brand in subdomain of unrelated domain
        return f"http://{b}.{word}-{random.randint(10, 999)}.{tld}/login"
    if style == 2:  # hyphenated look-alike
        return f"https://{b}-{word}-{random.choice(PHISH_WORDS)}.{tld}/?id={random.randint(1000, 99999)}"
    if style == 3:  # typosquat
        typo = b.replace("a", "4", 1).replace("o", "0", 1).replace("l", "1", 1)
        return f"http://www.{typo}.com/{word}/index.html"
    if style == 4:  # URL shortener
        code = "".join(random.choices("abcdefghijkmnpqrstuvwxyz23456789", k=6))
        return f"http://{random.choice(['bit.ly', 'tinyurl.com', 'is.gd'])}/{code}"
    return f"http://{word}.{b}.account-{random.randint(100, 999)}.{tld}/@{b}/verify"


def legit_url() -> str:
    domain = random.choice(LEGIT_DOMAINS)
    path = random.choice(["", "account", "orders", "help", "settings",
                          "docs/getting-started", "events/2024", "blog"])
    return f"https://www.{domain}/{path}".rstrip("/")


PHISHING_TEMPLATES = [
    "Dear Customer,\n\nWe detected unusual activity on your {brand} account. Your account has been "
    "temporarily suspended. To restore access, verify your identity immediately at {url}\n\n"
    "Failure to verify within 24 hours will result in permanent account closure.\n\n{brand} Security Team",
    "URGENT: Your {brand} password expires today! Click here to keep your current password: {url}\n"
    "Act now to avoid losing access to your account.",
    "Congratulations {name}! You have been selected as the winner of a $1,000 gift card. "
    "Claim your prize now before it expires: {url}\nLimited time offer!!!",
    "Hello,\n\nYour {brand} payment could not be processed. Please update your billing information "
    "and credit card details here: {url}\n\nIf you do not update within 48 hours your order will be cancelled.",
    "Your package from {brand} could not be delivered due to an unpaid shipping fee of $1.99. "
    "Pay now to reschedule delivery: {url}",
    "Security Alert: Someone tried to sign in to your {brand} account from an unknown device. "
    "If this wasn't you, confirm your account details immediately: {url}",
    "Dear valued user, we are upgrading our email servers. Login below to verify your mailbox or it "
    "will be deactivated: {url}\n\nIT Helpdesk",
    "{name}, you have a pending tax refund of $842.50. To receive your refund, submit your social "
    "security number and bank account details at {url}",
    "Hi {name},\n\nI need you to purchase gift cards for a client urgently. Send me the codes as soon "
    "as possible. I'm in a meeting so can't talk. Details here: {url}\n\nThanks, CEO",
    "Invoice #{num} is attached and overdue. Open the secure document to view the payment details: {url}\n"
    "Kindly confirm your login credentials to access the file.",
    "Hi {name}, a shared document is waiting for you on {brand}. Sign in with your email password to "
    "view it: {url}",
    "Your {brand} subscription is on hold. We were unable to validate your billing information. "
    "Please verify your account to continue enjoying our service: {url}",
    # Harder phishing: no link, polite tone, or a link that looks legitimate.
    "Hi {name}, this is IT support. We are migrating accounts this week. Please reply to this email with "
    "your username and current password so we can move your mailbox without interruption.",
    "Hello {name}, hope you are well. Our records show your {brand} profile is incomplete. When you have "
    "a moment please review it here: {url}\n\nThank you, Customer Care",
    "Hi {name}, the HR department has shared the updated salary review document with you. "
    "You will need to sign in to view it: {url}",
]

LEGIT_TEMPLATES = [
    "Hi {name},\n\nThanks for your order #{num}. It has shipped and should arrive on Thursday. "
    "You can track the package in your account: {url}\n\nThe {brand} team",
    "Hi team,\n\nReminder that our weekly sync is moved to 3pm tomorrow. Agenda: roadmap review, "
    "hiring update, and Q&A. Notes from last week are here: {url}\n\nBest, {name}",
    "Hey {name}, great seeing you at the conference! Here are the slides from my talk as promised: {url}. "
    "Let me know if you want to grab coffee next week.",
    "Your monthly statement for account ending in {short} is now available. Log in through our app or "
    "website as usual to view it. We will never ask for your password by email.",
    "Hi {name},\n\nAttached is the draft report for the project. Could you review sections 2 and 3 "
    "before Friday? The shared folder is at {url}\n\nThanks!",
    "Welcome to the newsletter! This month: five tips for better sleep, a recipe for lentil soup, and "
    "an interview with our community lead. Read online: {url}",
    "Hi {name}, your appointment with Dr. Patel is confirmed for Monday at 10:30am. Reply to this "
    "email if you need to reschedule.",
    "The library books you borrowed are due next week. You can renew them online at {url} or visit "
    "the front desk.",
    "Hello {name},\n\nWe noticed a new sign-in to your {brand} account from Chrome on Windows. If this "
    "was you, no action is needed. You can review your security settings at any time from your account page.",
    "Hi {name}, just a reminder that the team lunch is on Friday at noon. Please reply with any dietary "
    "restrictions by Wednesday.",
    "Your pull request #{num} was merged. Thanks for contributing! View the changes: {url}",
    "Hi {name},\n\nThe receipt for your {brand} purchase of $12.99 is below. No action is required. "
    "Questions? Visit our help centre at {url}",
    "Dear {name},\n\nThank you for applying to the position. We would like to invite you to an "
    "interview next Tuesday. Please let us know which time slot works for you.\n\nKind regards, HR",
    "Mom says dinner is at 7 on Sunday, bring the photos from the trip! Also here's the recipe link: {url}",
    # Harder legitimate emails: urgency, passwords or account wording used innocently.
    "URGENT: the production server is down. Please join the incident call immediately. Status page: {url}",
    "Hi {name}, you requested a password reset for your {brand} account. If you did not make this "
    "request you can ignore this email. Reset link (expires in 1 hour): {url}",
    "Reminder: please confirm your attendance for Friday's workshop by replying to this email. "
    "Limited seats available!",
    "Hi {name}, congratulations on your promotion! The whole team is really proud of you. Drinks on Friday?",
    "Your {brand} account settings were updated. To verify this change was made by you, review your "
    "recent activity in the app. Old site links like {url} still work.",
]


def make_email(label: int) -> str:
    fields = {
        "name": random.choice(NAMES),
        "brand": random.choice(BRANDS),
        "num": random.randint(10000, 999999),
        "short": random.randint(1000, 9999),
    }
    if label == 1:
        # Some attackers host pages on legitimate-looking HTTPS domains.
        fields["url"] = legit_url() if random.random() < 0.1 else phishing_url(fields["brand"])
        text = random.choice(PHISHING_TEMPLATES).format(**fields)
        if random.random() < 0.3:
            text = text.upper() if random.random() < 0.3 else text + "\n\nDo not ignore this message!"
    else:
        # Legitimate mail sometimes contains plain-HTTP or shortened links too.
        fields["url"] = legit_url()
        if random.random() < 0.1:
            fields["url"] = random.choice([fields["url"].replace("https://", "http://"),
                                           "https://bit.ly/" + str(random.randint(10000, 99999))])
        text = random.choice(LEGIT_TEMPLATES).format(**fields)
    return text


def generate(n: int, seed: int = 42) -> list[tuple[str, int]]:
    random.seed(seed)
    rows = [(make_email(label), label) for label in (random.randint(0, 1) for _ in range(n))]
    # Label noise (~2%) mimics mislabelled real-world data so accuracy is realistic.
    return [(t, 1 - l) if random.random() < 0.02 else (t, l) for t, l in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=2000, help="number of emails to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "emails.csv")
    args = parser.parse_args()

    rows = generate(args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for text, label in rows:
            writer.writerow([text, "Phishing" if label else "Safe"])
    print(f"Wrote {len(rows)} emails to {args.out}")


if __name__ == "__main__":
    main()
