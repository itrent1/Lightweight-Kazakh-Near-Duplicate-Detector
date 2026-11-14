"""Named SQL report queries (JOIN, GROUP BY/HAVING, subqueries, CTE, window functions)."""

QUERIES = {
    "type_summary": {
        "title": "Q1. Duplicate types: size, mean score and length (GROUP BY)",
        "sql": """
            SELECT type_duplicate,
                   COUNT(*)                          AS pairs,
                   ROUND(AVG(similarity_score), 3)   AS avg_score,
                   ROUND(AVG(content_len_tokens), 1) AS avg_content_len
            FROM pairs
            GROUP BY type_duplicate
            ORDER BY avg_score DESC
        """,
    },
    "accuracy_by_type": {
        "title": "Q2. Type accuracy per method and type (JOIN + GROUP BY)",
        "sql": """
            SELECT e.method,
                   p.type_duplicate,
                   COUNT(*) AS n,
                   ROUND(AVG(CASE WHEN pr.pred_type = p.type_duplicate THEN 1.0 ELSE 0.0 END), 3) AS accuracy
            FROM predictions pr
            JOIN pairs p       ON p.pair_id = pr.pair_id
            JOIN experiments e ON e.experiment_id = pr.experiment_id
            GROUP BY e.method, p.type_duplicate
            ORDER BY e.method, accuracy
        """,
    },
    "methods_above_threshold": {
        "title": "Q3. Methods whose overall accuracy is above 0.3 (GROUP BY + HAVING)",
        "sql": """
            SELECT e.method,
                   COUNT(*) AS pairs,
                   ROUND(AVG(CASE WHEN pr.pred_type = p.type_duplicate THEN 1.0 ELSE 0.0 END), 3) AS accuracy
            FROM predictions pr
            JOIN pairs p       ON p.pair_id = pr.pair_id
            JOIN experiments e ON e.experiment_id = pr.experiment_id
            GROUP BY e.method
            HAVING AVG(CASE WHEN pr.pred_type = p.type_duplicate THEN 1.0 ELSE 0.0 END) > 0.3
            ORDER BY accuracy DESC
        """,
    },
    "confusion_counts": {
        "title": "Q4. Confusion counts for the chosen method (JOIN + GROUP BY)",
        "sql": """
            SELECT p.type_duplicate AS true_type,
                   pr.pred_type     AS predicted_type,
                   COUNT(*)         AS n
            FROM predictions pr
            JOIN pairs p       ON p.pair_id = pr.pair_id
            JOIN experiments e ON e.experiment_id = pr.experiment_id
            WHERE e.method = :method
            GROUP BY true_type, predicted_type
            ORDER BY n DESC
            LIMIT 20
        """,
    },
    "largest_score_errors": {
        "title": "Q5. Pairs with the largest score error (JOIN + ORDER BY)",
        "sql": """
            SELECT p.pair_id,
                   p.type_duplicate,
                   pr.pred_type,
                   p.similarity_score,
                   ROUND(pr.pred_score, 3) AS pred_score,
                   ROUND(ABS(pr.pred_score - p.similarity_score), 3) AS abs_error
            FROM predictions pr
            JOIN pairs p       ON p.pair_id = pr.pair_id
            JOIN experiments e ON e.experiment_id = pr.experiment_id
            WHERE e.method = :method
            ORDER BY abs_error DESC
            LIMIT 20
        """,
    },
    "longer_than_type_average": {
        "title": "Q6. Pairs longer than the average of their type (correlated subquery)",
        "sql": """
            SELECT p.type_duplicate,
                   COUNT(*) AS longer_than_avg
            FROM pairs p
            WHERE p.content_len_tokens IS NOT NULL
              AND p.content_len_tokens > (
                SELECT AVG(q.content_len_tokens)
                FROM pairs q
                WHERE q.type_duplicate = p.type_duplicate
                  AND q.content_len_tokens IS NOT NULL
            )
            GROUP BY p.type_duplicate
            ORDER BY longer_than_avg DESC
        """,
    },
    "hardest_pairs_per_type": {
        "title": "Q7. Top-3 largest errors inside each type (window ROW_NUMBER)",
        "sql": """
            SELECT * FROM (
                SELECT p.type_duplicate,
                       p.pair_id,
                       ROUND(ABS(pr.pred_score - p.similarity_score), 3) AS abs_error,
                       ROW_NUMBER() OVER (
                           PARTITION BY p.type_duplicate
                           ORDER BY ABS(pr.pred_score - p.similarity_score) DESC
                       ) AS rn
                FROM predictions pr
                JOIN pairs p       ON p.pair_id = pr.pair_id
                JOIN experiments e ON e.experiment_id = pr.experiment_id
                WHERE e.method = :method
            )
            WHERE rn <= 3
            ORDER BY type_duplicate, rn
        """,
    },
    "type_share_ranking": {
        "title": "Q8. Share and rank of each duplicate type (window SUM OVER + RANK)",
        "sql": """
            SELECT type_duplicate,
                   COUNT(*) AS pairs,
                   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS share_pct,
                   RANK() OVER (ORDER BY COUNT(*) DESC) AS size_rank
            FROM pairs
            GROUP BY type_duplicate
            ORDER BY size_rank
        """,
    },
    "accuracy_by_length": {
        "title": "Q9. Accuracy by document length bucket (CTE + CASE)",
        "sql": """
            WITH scored AS (
                SELECT CASE
                           WHEN p.content_len_tokens < 100 THEN '1: <100'
                           WHEN p.content_len_tokens < 300 THEN '2: 100-299'
                           WHEN p.content_len_tokens < 600 THEN '3: 300-599'
                           ELSE '4: 600+'
                       END AS length_bucket,
                       CASE WHEN pr.pred_type = p.type_duplicate THEN 1.0 ELSE 0.0 END AS correct
                FROM predictions pr
                JOIN pairs p       ON p.pair_id = pr.pair_id
                JOIN experiments e ON e.experiment_id = pr.experiment_id
                WHERE e.method = :method
                  AND p.content_len_tokens IS NOT NULL
            )
            SELECT length_bucket,
                   COUNT(*)                    AS pairs,
                   ROUND(AVG(correct), 3)      AS accuracy
            FROM scored
            GROUP BY length_bucket
            ORDER BY length_bucket
        """,
    },
    "method_leaderboard": {
        "title": "Q10. Method leaderboard by macro-F1 (CTE + RANK window)",
        "sql": """
            WITH latest AS (
                SELECT e.*
                FROM experiments e
                WHERE e.experiment_id IN (SELECT MAX(experiment_id) FROM experiments GROUP BY method)
            )
            SELECT method,
                   ROUND(f1_macro, 4)               AS f1_macro,
                   ROUND(spearman, 4)               AS spearman,
                   ROUND(recall_at_10, 4)           AS recall_at_10,
                   ROUND(infer_seconds_per_1000, 3) AS sec_per_1000_pairs,
                   ROUND(model_size_mb, 1)          AS size_mb,
                   RANK() OVER (ORDER BY f1_macro DESC) AS f1_rank
            FROM latest
            ORDER BY f1_rank
        """,
    },
}
