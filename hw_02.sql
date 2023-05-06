USE baseball;

CREATE TABLE historic_batting_avg AS
SELECT batter, SUM(hit) / SUM(atbat) AS historic_average
FROM batter_counts
WHERE atbat > 0 
GROUP BY batter
;

CREATE TABLE annual_batting_avg AS
SELECT batter, YEAR(updateddate) AS bat_year, SUM(hit) / SUM(atbat) AS annual_average
FROM batter_counts
WHERE atbat > 0 
GROUP BY batter, bat_year
;

CREATE TEMPORARY TABLE prior_data (SELECT local_date, batter, hit, atbat
  FROM batter_counts
  JOIN game ON game.game_id = batter_counts.game_id);

 
CREATE TABLE rolling_batting_avg AS
SELECT a.batter, a.local_date,
	CASE WHEN SUM(b.atbat)>0 THEN SUM(b.hit) / SUM(b.atbat)
	ELSE 0
	END AS rolling_average
FROM prior_data AS a 
JOIN prior_data AS b 
       ON a.batter = b.batter
            AND b.local_date >= a.local_date - INTERVAL '100' DAY
            AND b.local_date < a.local_date
GROUP BY a.batter, a.local_date;

