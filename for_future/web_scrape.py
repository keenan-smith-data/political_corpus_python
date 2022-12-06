from bs4 import BeautifulSoup
import requests
import numpy as np
import pandas as pd
import regex as re

def link_getting(url, my_headers, explore = True, html_get = "a", reg_exp = "", parser = "html.parser"):

    assert isinstance(url, str), f"url must be a String"
    assert isinstance(reg_exp, str), f"Regular Expression must be a String"

    session = requests.Session()
    try:
        con = session.get(url, headers = my_headers)
    except Exception as e:
        print("Error feteching the URL ", url)
        print(e)
    try:
        soup = BeautifulSoup(con.text, parser)
    except Exception as e:
        print("Could not parse: ", url)
        print(e)
    if explore:
        container_list = soup.find_all(html_get)
        return([tag for tag in container_list])
    else:
        container_list = soup.find_all(html_get, href=re.compile(reg_exp))
        return([tag.get("href") for tag in container_list])
    
def unique_array(list1):

    assert isinstance(list1, list), f"List1 must be a list"

    temp = np.array(list1)
    return np.unique(temp)

def link_cleaning(list1, reg_exp = ""):

    assert isinstance(list1, list), f"List1 must be a list"
    assert isinstance(reg_exp, str), f"Regular Expression must be a String"

    temp = np.char.strip(unique_array(list1))
    regex = re.compile(reg_exp)
    return [tag for tag in temp if regex.match(tag)]