import os
import requests


def scrape_linkedin_profile(linkedin_profile_url: str):
    """scrape information from LinkedIn profiles,
    Manually scrape the information from the LinkedIn profile"""
    # api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
    # header_dic = {"Authorization": f'Bearer {os.environ.get("PROXYCURL_API_KEY")}'}
    #
    # response = requests.get(
    #     api_endpoint, params={"url": linkedin_profile_url}, headers=header_dic
    # )
    api_endpoint = "https://gist.githubusercontent.com/gkjegan/d7d52076bd4f43005308410a951ba569/raw/3e634dbd1859f188319232a55ab4ea0ffce4b61c/gkjegan.json"
    response = requests.get(api_endpoint)
    data = response.json()
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "", None)
        and k not in ["people_also_viewed", "certifications"]
    }
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")
    return data
