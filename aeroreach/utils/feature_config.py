# Configuration for feature columns

CATEGORICAL_COLS = [
    'Taken_product',
    'preferred_device',
    'member_in_family',
    'preferred_location_type',
    'following_company_page',
    'working_flag',
    'Adult_flag(Age Group)'
]

NUMERICAL_COLS = [
    'Yearly_avg_view_on_travel_page',
    'total_likes_on_outstation_checkin_given',
    'yearly_avg_Outstation_checkins',
    'Yearly_avg_comment_on_travel_page',
    'total_likes_on_outofstation_checkin_received',
    'week_since_last_outstation_checkin',
    'montly_avg_comment_on_company_page',
    'travelling_network_rating',
    'Daily_Avg_mins_spend_on_traveling_page'
]

TARGET_COL = 'Taken_product'
