import requests
import json
import pandas as pd 
import time
# get Bearer Token
def getToken(proxy_url, tenant_id, client_id, client_secret, resourceUrl,baseAuthUrl):
    proxies = {'http': proxy_url, 'https': proxy_url}
    auth_url = baseAuthUrl.format(tenant_id) 
    body={'resource':resourceUrl, 
          'grant_type':'client_credentials', 
          'client_id':client_id, 
          'client_secret':client_secret}
    result = requests.post(auth_url,proxies=proxies,data=body)
    jsonRes=json.loads(result.text)
    accessToken=jsonRes["access_token"]    
    return accessToken

# get Group Data
def getGroups(tenant_id, proxy_url, accessToken, resourceUrl):
    URL = resourceUrl+tenant_id + "/groups?api-version=1.6"
    proxies = {'http': proxy_url, 'https': proxy_url}
    token = "Bearer "+accessToken 
    headers  = {'Authorization':token, 'content-type': 'application/json'} 
    r = requests.get(url = URL, proxies=proxies, headers = headers ) 
    data = r.json()
    return data

# add Group
def addGroup(proxy_url, tenant_id, accessToken, resourceUrl,displayName,mailNickname,mailEnabled, securityEnabled):
    URL = resourceUrl+tenant_id + "/groups?api-version=1.6"
    token = "Bearer "+accessToken 
    headers  = {'Authorization':token, 'content-type': 'application/json'}
    proxies = {'http': proxy_url, 'https': proxy_url}
    body={
            "displayName": displayName,
            "mailNickname": mailNickname,
            "mailEnabled": mailEnabled,
            "securityEnabled": securityEnabled
            }
    result = requests.post(URL,data=json.dumps(body),proxies=proxies, headers = headers)
    jsonRes=json.loads(result.text)    
    return jsonRes

# add user to Group
def addUserToGroup(proxy_url, tenant_id, accessToken, resourceUrl, userId, groupId):
    URL = resourceUrl+tenant_id + "/groups/"+groupId+ "/$links/members?api-version=1.6"
    token = "Bearer "+accessToken 
    headers  = {'Authorization':token, 'content-type': 'application/json'}
    proxies = {'http': proxy_url, 'https': proxy_url}
    userUrl="https://graph.windows.net/" + tenant_id+ "/directoryObjects/"+userId
    body={
            "url": userUrl
            }
    result = requests.post(URL,data=json.dumps(body),proxies=proxies, headers = headers)
    return result
# get user
def getUser(proxy_url, tenant_id, accessToken, resourceUrl, userId):
    URL = resourceUrl+tenant_id + "/users/"+userId+ "?api-version=1.6"    
    proxies = {'http': proxy_url, 'https': proxy_url}
    token = "Bearer "+accessToken 
    headers  = {'Authorization':token, 'content-type': 'application/json'} 
    r = requests.get(url = URL, proxies=proxies, headers = headers ) 
    data = r.json()
    return data
# read Group Dictionary
def loadGroupData(groupData):
    groupNames = {}
    for i in groupData['value']:
        groupNames[i['displayName']]=i['objectId']
    return groupNames
# add user to Group
def addUser(proxy_url, tenant_id, accessToken, resourceUrl, userId, accountEnabled, displayUserName, mailUserNickname,password,forceChangePasswordNextLogin):
    URL = resourceUrl+tenant_id + "/users?api-version=1.6"
    token = "Bearer "+accessToken 
    headers  = {'Authorization':token, 'content-type': 'application/json'}
    proxies = {'http': proxy_url, 'https': proxy_url}    
    body={
      "accountEnabled": accountEnabled,
      "displayName": displayUserName,
      "mailNickname": mailUserNickname,
      "passwordProfile": {
        "password": password,
        "forceChangePasswordNextLogin": forceChangePasswordNextLogin
      },
      "userPrincipalName": userId
    }
    result = requests.post(URL,data=json.dumps(body),proxies=proxies, headers = headers)
    return result


# load Groups
def loadGroups(tenant_id, proxy_url, accessToken, resourceUrl):
    groupData=getGroups(tenant_id, proxy_url, accessToken, resourceUrl)    
    groupNamesTemp=loadGroupData(groupData)
    return groupNamesTemp
data = pd.read_csv("userDataLoad.csv") 
tokenList={}
groupNames={}
groupData={}
res={}
resAddRes={}
addGroupRes={}
reload=True
for i in range(0,len(data)):
    proxy_url = str(data["proxy_url"][i])
    tenant_id = str(data["tenant_id"][i])
    client_id = str(data["client_id"][i])
    client_secret = str(data["client_secret"][i])
    baseAuthUrl=str(data["baseAuthUrl"][i])
    resourceUrl=str(data["resourceUrl"][i])
    displayName=str(data["displayName"][i])
    mailNickname=str(data["mailNickname"][i])
    mailEnabled=str(data["mailEnabled"][i]).lower()
    securityEnabled=str(data["securityEnabled"][i]).lower()
    accountEnabled=str(data["accountEnabled"][i]).lower()
    forceChangePasswordNextLogin=str(data["forceChangePasswordNextLogin"][i]).lower()
    user=str(data["userPrincipalName"][i])
    displayUserName=str(data["displayUserName"][i])
    mailUserNickname=str(data["mailUserNickname"][i])
    password=str(data["password"][i])
    accessToken=tokenList.get(client_id, 0)
    if accessToken== 0:
        accessToken=getToken(proxy_url, tenant_id, client_id, client_secret, resourceUrl,baseAuthUrl)
        tokenList[client_id]=accessToken
    if reload:        
        groupNames=loadGroups(tenant_id, proxy_url, accessToken, resourceUrl)
        reload=False
    groupId=groupNames.get(displayName, 0)
    if groupId== 0:
        addGroupRes=addGroup(proxy_url, tenant_id, accessToken, resourceUrl,displayName,mailNickname,mailEnabled, securityEnabled)    
        groupId=addGroupRes['objectId']
        print('CREATED --- Group Name-', displayName, ' Group Id -', groupId)        
        reload=True        
        time.sleep(15) 
    else:
        print('EXISTING --- Group Name-', displayName, ' Group Id -', groupId)
    res=getUser(proxy_url, tenant_id, accessToken, resourceUrl, user)
    userObjectId=res.get('objectId', 0)
    if userObjectId== 0:
        resAddRes=addUser(proxy_url, tenant_id, accessToken, resourceUrl, user, accountEnabled, displayUserName, mailUserNickname,password,forceChangePasswordNextLogin)        
        userObjectId=resAddRes.json()['objectId']
        print('CREATED --- User Name-', user, ' User Id -', userObjectId)
    else:
        print('EXISTING --- User Name-', user, ' User Id -', userObjectId)
    addUserToGroup(proxy_url, tenant_id, accessToken, resourceUrl, userObjectId, groupId)
    