class Tweet:
    #     user_screen_name = None
    #     user_name = None
    #     user_description = None
    #     user_followers_count = None
    #     user_friends_count = None
    #     user_listed_count = None
    #     user_favourites_count = None
    #     user_isVerified = None
    #     user_statuses_count = None
    #     user_location = None
    #     user_created_at = None
    #     user_geo_enabled = None

    #     tweet_created_at = None
    #     tweet_geo = None
    #     tweet_coordinates = None
    #     tweet_place = None
    #     tweet_contributors = None
    #     tweet_is_quote_status = None
    #     tweet_retweet_count = None
    #     tweet_favorite_count = None
    #     tweet_isFavorited = None
    #     tweet_isRetweeted = None
    #     tweet_text = None
    #     tweet_polarity = None

    def __init__(self, json_response):
        self.user_screen_name = json_response.user.screen_name
        self.user_name = json_response.user.name
        self.user_description = json_response.user.description
        self.user_followers_count = json_response.user.followers_count
        self.user_friends_count = json_response.user.friends_count
        self.user_listed_count = json_response.user.listed_count
        self.user_favourites_count = json_response.user.favourites_count
        self.user_isVerified = json_response.user.verified
        self.user_statuses_count = json_response.user.statuses_count
        self.user_location = json_response.user.location
        self.user_created_at = json_response.user.created_at
        self.user_geo_enabled = json_response.user.geo_enabled

        self.tweet_created_at = json_response.created_at
        self.tweet_geo = json_response.geo
        self.tweet_coordinates = json_response.coordinates
        self.tweet_place = json_response.place
        self.tweet_contributors = json_response.contributors
        self.tweet_is_quote_status = json_response.is_quote_status
        self.tweet_retweet_count = json_response.retweet_count
        self.tweet_favorite_count = json_response.favorite_count
        self.tweet_isFavorited = json_response.favorited
        self.tweet_isRetweeted = json_response.retweeted
        self.tweet_text = json_response.text








