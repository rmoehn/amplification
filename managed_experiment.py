import argparse
import ast
import email.message
import os
import smtplib
import sys

import sh


def send_notification(to: str, subject: str, body: str):
    with open(os.getenv("MAILPASS_PATH"), "r") as f:
        password = f.read().rstrip()
    with open(os.getenv("MAILCONFIG_PATH"), "r") as f:
        config = ast.literal_eval(f.read())

    msg = email.message.EmailMessage()
    msg["Subject"] = subject
    msg["From"] = config["from_address"]
    msg["To"] = to
    msg.set_content(body)

    smtp = smtplib.SMTP(config["server"], config["port"])

    response = smtp.starttls()
    # Make it more secure on older Pythons.
    # See https://stackoverflow.com/a/33891521/5091738.
    if response[0] != 220:
        raise RuntimeError("Failed to set up STARTTLS.")

    smtp.ehlo()
    smtp.login(config["username"], password)
    smtp.send_message(msg)
    smtp.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--receiver", required=True)
    parser.add_argument("to_run")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    python = sh.Command(sys.executable)
    try:
        python(args.to_run, *args.args, _fg=True)
    except sh.ErrorReturnCode:
        send_notification(to=args.receiver, subject="AMPLIFICATION Failed", body="")
