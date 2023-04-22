USE baseball;

CREATE TABLE features AS
WITH cte_game_stats AS (
    SELECT
        b.game_id
        , b.batter
        , b.atBat
        , b.Hit
        , b.toBase
        , b.Home_Run
        , b.plateApperance
        , b.homeTeam AS batter_home
        , b.awayTeam  AS batter_away
        , p.pitcher
        , p.Strikeout
        , p.Walk
        , p.Ground_Out
        , p.Fly_Out
        , p.homeTeam AS pitcher_home
        , p.awayTeam AS pitcher_away
        , p.pitchesThrown
        , p.DaysSinceLastPitch
        , p.endingInning - p.startingInning AS inningsPitched
    FROM
        batter_counts AS b
        JOIN pitcher_counts AS p ON b.game_id = p.game_id
)
SELECT
    cte.game_id
    , cte.batter
    , cte.pitcher
    , cte.batter_home
    , cte.batter_away
    , cte.pitcher_home
    , cte.pitcher_away
    , cte.Home_Run / NULLIF(cte.Hit, 0) AS hr_h
    , cte.toBase / NULLIF(cte.atBat, 0) AS obp
    , cte.plateApperance / NULLIF(cte.Strikeout, 0) AS pa_so
    , cte.WALK / 9 AS bb_9
    , cte.Hit / 9 AS h_9
    , cte.Home_Run / 9 AS hr_9
    , cte.Strikeout / 9 AS k_9
    , cte.Ground_Out / NULLIF(cte.Fly_Out, 0) AS go_ao
    , cte.Strikeout / NULLIF(cte.pitchesThrown, 0) AS k_pitch_load
    , cte.Strikeout / NULLIF(cte.DaysSinceLastPitch, 0) AS k_rest
    , boxscore.winner_home_or_away
FROM
    cte_game_stats AS cte
    JOIN boxscore ON boxscore.game_id = cte.game_id
;
