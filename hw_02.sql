SELECT batter, SUM(hit) / SUM(atbat) AS historic_batting_average
FROM batter_counts
GROUP BY batter
;

SELECT batter, YEAR(updateddate) AS bat_year, SUM(hit) / SUM(atbat) AS annual_batting_average
FROM batter_counts
GROUP BY batter, bat_year
;

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
GROUP BY
    a.batter
    , a.updateddate
;
