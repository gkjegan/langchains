import requests
import json

api_key = "AZTo2ZkcXp2JFF0MR7tGtQ"
headers = {"Authorization": "Bearer " + api_key}
api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
params = {
    "linkedin_profile_url": "https://linkedin.com/in/johnrmarty/",
    "extra": "include",
    "github_profile_id": "include",
    "facebook_profile_id": "include",
    "twitter_profile_id": "include",
    "personal_contact_number": "include",
    "personal_email": "include",
    "inferred_salary": "include",
    "skills": "include",
    "use_cache": "if-present",
    "fallback_to_cache": "on-error",
}
response = requests.get(api_endpoint, params=params, headers=headers)

print(response.json())
data = response.json()
with open("../data/data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
