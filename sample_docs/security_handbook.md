# Northwind Information Security Handbook

Version 4.2. Classification: Internal.

## Passwords and authentication

All employees must use the company password manager. Passwords must be at least 14 characters long and must not be reused across systems. Multi-factor authentication (MFA) is mandatory for email, the VPN, the source code platform and all cloud consoles. Hardware security keys are required for anyone with administrator access to production.

## Reporting a security incident

Any suspected security incident, such as a lost laptop, a phishing email that was clicked, or unexpected account activity, must be reported to the Security Operations Center within one hour of discovery. Report by emailing soc@northwind.example or by calling the 24-hour hotline on extension 5100. Do not attempt to investigate or delete evidence yourself.

## Data classification

Northwind uses four data classes: Public, Internal, Confidential and Restricted. Customer personal data and payment data are Restricted. Restricted data must be encrypted at rest with AES-256 and must never be copied to personal devices or sent through consumer messaging apps. Confidential data may be shared externally only under a signed non-disclosure agreement.

## Software and devices

Only software from the approved catalogue may be installed on company laptops. Requests for new software go through the IT Service Desk, and a security review takes up to 10 business days. Laptops are encrypted with full-disk encryption and lock automatically after five minutes of inactivity. Lost or stolen devices can be remotely wiped by IT.

## Access reviews

Managers review the access rights of their team every quarter. Access of departing employees is revoked on their last working day by 17:00. Contractors receive access that expires automatically after 90 days unless renewed.

## Security training

All employees complete security awareness training within 30 days of joining and again every year. Engineers additionally complete secure coding training. Simulated phishing campaigns run every quarter.
