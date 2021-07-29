import tweepy

class Request:

    def __init__(self):
        self.consumer_key = None
        self.consumer_secret = None
        self.access_token = None
        self.access_token_secret = None
        self.getApiKeys()

    def getApiKeys(self):
        keys_file = open("keys.txt")
        lines = keys_file.readlines()
        consumer_key = lines[0].rstrip()
        consumer_secret = lines[1].rstrip()
        access_token = lines[2].rstrip()
        access_token_secret = lines[3].rstrip()

        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret


    def requestApiAccess(self):
        auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
        auth.set_access_token(self.access_token, self.access_token_secret)
        api = tweepy.API(auth)

        return api

