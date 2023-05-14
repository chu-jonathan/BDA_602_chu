USE baseball;

DROP TABLE IF EXISTS betting_odds;
CREATE TEMPORARY TABLE betting_odds AS
SELECT tr.game_id, tr.team_id, tr.local_date, tr.home_away, tr.win_lose, tr.streak, po.home_team_id, MIN(po.home_line) AS first_home_line
FROM team_results AS tr
    LEFT JOIN pregame_odds AS po ON tr.game_id = po.game_id
WHERE tr.home_away = 'H'
GROUP BY game_id
;

DROP TABLE IF EXISTS pitching_raw;
CREATE TEMPORARY TABLE pitching_raw AS
SELECT bo.game_id, bo.team_id, bo.local_date, bo.home_away, bo.win_lose, bo.streak, bo.home_team_id, bo.first_home_line
    , pc.pitcher, pc.startingPitcher, pc.homeTeam, pc.Strikeout, pc.Walk, pc.Ground_Out, pc.Fly_Out, pc.pitchesThrown
    , pc.DaysSinceLastPitch, pc.endingInning - pc.startingInning AS inningsPitched
FROM betting_odds bo
    LEFT JOIN pitcher_counts pc ON bo.game_id = pc.game_id
WHERE pc.startingPitcher = 1 AND pc.homeTeam = 1
;

DROP TABLE IF EXISTS pitching_stats;
CREATE TABLE pitching_stats AS
SELECT game_id, team_id, local_date, home_away, win_lose, streak, home_team_id, first_home_line
	, pitcher, startingPitcher, homeTeam, Strikeout, Walk, Ground_Out, Fly_Out
	, pitchesThrown, DaysSinceLastPitch, inningsPitched, WALK / 9 AS bb_9, Strikeout / 9 AS k_9
    , COALESCE(GROUND_OUT / NULLIF(Fly_Out, 0), 0) AS go_ao
    , COALESCE(Strikeout / NULLIF(pitchesThrown, 0), 0) AS k_pitch_load
    , COALESCE(Strikeout / NULLIF(DaysSinceLastPitch, 0), 0) AS k_rest
    , MONTH(local_date) AS month_column
    , COALESCE(DATEDIFF(local_date, LAG(local_date) OVER (PARTITION BY team_id ORDER BY local_date)), 0) AS days_since_last_game
FROM pitching_raw
;
