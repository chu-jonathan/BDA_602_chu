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

CREATE TABLE rolling_batting_avg AS
SELECT
    a.batter
    , a.updateddate
    , AVG(b.hit / b.atbat) AS rolling_average
FROM
    batter_counts AS a
    JOIN batter_counts AS b
        ON a.batter = b.batter
            AND b.updateddate >= a.updateddate - INTERVAL '100' DAY
            AND b.updateddate < a.updateddate
            AND b.atbat > 0
WHERE a.atbat > 0
GROUP BY
    a.batter
    , a.updateddate
;
