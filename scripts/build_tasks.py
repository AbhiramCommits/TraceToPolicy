"""Author the hand-written NL->SQL task corpus and write ``data/tasks.jsonl``.

Each task is a dict with ``task_id``, ``question``, ``gold_sql`` and
``category``. Categories span simple filtering, joins, group-by/having, window
functions, date arithmetic, nested subqueries and ambiguous-column cases.

Run: python scripts/build_tasks.py
"""

from __future__ import annotations

import json
import os

T = []


def add(question: str, sql: str, category: str) -> None:
    T.append(
        {"question": question.strip(), "gold_sql": " ".join(sql.split()), "category": category}
    )


# --------------------------------------------------------------------------- #
# Simple filtering / projection
# --------------------------------------------------------------------------- #
add(
    "List the names of all customers in the US.",
    "SELECT name FROM customers WHERE country = 'US'",
    "filter",
)
add(
    "How many products cost more than 100?",
    "SELECT COUNT(*) FROM products WHERE price > 100",
    "filter",
)
add(
    "What are the distinct countries customers come from?",
    "SELECT DISTINCT country FROM customers",
    "filter",
)
add(
    "List product names and prices in the electronics category, most expensive first.",
    "SELECT name, price FROM products WHERE category = 'electronics' ORDER BY price DESC",
    "filter",
)
add(
    "Find the email addresses of enterprise-segment customers.",
    "SELECT email FROM customers WHERE segment = 'enterprise'",
    "filter",
)
add(
    "Count the number of cancelled orders.",
    "SELECT COUNT(*) FROM orders WHERE status = 'cancelled'",
    "filter",
)
add(
    "List the five most expensive products.",
    "SELECT name, price FROM products ORDER BY price DESC LIMIT 5",
    "filter",
)
add(
    "How many events were recorded on iOS devices?",
    "SELECT COUNT(*) FROM events WHERE device = 'ios'",
    "filter",
)
add("What distinct subscription plans exist?", "SELECT DISTINCT plan FROM subscriptions", "filter")
add(
    "Find all products whose brand is 'Acme'.",
    "SELECT name FROM products WHERE brand = 'Acme'",
    "filter",
)
add(
    "List order ids placed through the mobile_app channel.",
    "SELECT order_id FROM orders WHERE channel = 'mobile_app'",
    "filter",
)
add(
    "How many customers are in the 'smb' segment?",
    "SELECT COUNT(*) FROM customers WHERE segment = 'smb'",
    "filter",
)
add(
    "What is the average list price of all products?",
    "SELECT AVG(price) FROM products",
    "aggregate",
)
add(
    "What is the highest single order total amount?",
    "SELECT MAX(total_amount) FROM orders",
    "aggregate",
)
add(
    "What is the total revenue across all completed orders?",
    "SELECT SUM(total_amount) FROM orders WHERE status = 'completed'",
    "aggregate",
)
add(
    "How many active subscriptions are there?",
    "SELECT COUNT(*) FROM subscriptions WHERE status = 'active'",
    "filter",
)
add(
    "List the names and categories of the 10 cheapest products.",
    "SELECT name, category FROM products ORDER BY price ASC LIMIT 10",
    "filter",
)
add(
    "How many orders were placed via the partner channel?",
    "SELECT COUNT(*) FROM orders WHERE channel = 'partner'",
    "filter",
)
add("What is the minimum product cost?", "SELECT MIN(cost) FROM products", "aggregate")
add("List distinct event types.", "SELECT DISTINCT event_type FROM events", "filter")

# --------------------------------------------------------------------------- #
# Joins
# --------------------------------------------------------------------------- #
add(
    "List each order id together with the name of the customer who placed it.",
    "SELECT o.order_id, c.name FROM orders o JOIN customers c ON o.customer_id = c.customer_id",
    "join",
)
add(
    "Show order id, order total and customer name for the 20 largest orders.",
    "SELECT o.order_id, o.total_amount, c.name FROM orders o JOIN customers c ON o.customer_id = c.customer_id ORDER BY o.total_amount DESC LIMIT 20",
    "join",
)
add(
    "List product names and quantities for order id 100.",
    "SELECT p.name, oi.quantity FROM order_items oi JOIN products p ON oi.product_id = p.product_id WHERE oi.order_id = 100",
    "join",
)
add(
    "For each customer, show their name and the number of orders they placed.",
    "SELECT c.name, COUNT(o.order_id) AS order_count FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.name",
    "join",
)
add(
    "List each product name with the total quantity sold.",
    "SELECT p.name, SUM(oi.quantity) AS units_sold FROM products p JOIN order_items oi ON p.product_id = oi.product_id GROUP BY p.name",
    "join",
)
add(
    "Show customer name and subscription plan for all active subscriptions.",
    "SELECT c.name, s.plan FROM subscriptions s JOIN customers c ON s.customer_id = c.customer_id WHERE s.status = 'active'",
    "join",
)
add(
    "List the names of customers who have never placed an order.",
    "SELECT c.name FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id WHERE o.order_id IS NULL",
    "join",
)
add(
    "For each order, show the order id and the number of distinct products in it.",
    "SELECT oi.order_id, COUNT(DISTINCT oi.product_id) AS n_products FROM order_items oi GROUP BY oi.order_id",
    "join",
)
add(
    "List the total revenue per customer segment.",
    "SELECT c.segment, SUM(o.total_amount) AS revenue FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.segment",
    "join",
)
add(
    "Show each product category and the number of products sold in it.",
    "SELECT p.category, COUNT(oi.item_id) AS items_sold FROM products p JOIN order_items oi ON p.product_id = oi.product_id GROUP BY p.category",
    "join",
)
add(
    "List the names of customers from Germany along with their order totals.",
    "SELECT c.name, o.total_amount FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE c.country = 'DE'",
    "join",
)
add(
    "For each subscription, show the customer name and the monthly recurring revenue.",
    "SELECT c.name, s.mrr FROM subscriptions s JOIN customers c ON s.customer_id = c.customer_id",
    "join",
)
add(
    "Show each order id with the product name and line revenue (quantity times unit price minus discount).",
    "SELECT oi.order_id, p.name, oi.quantity * oi.unit_price * (1 - oi.discount) AS line_revenue FROM order_items oi JOIN products p ON oi.product_id = p.product_id",
    "join",
)
add(
    "Which customers have more than 5 orders? Show their names and order counts.",
    "SELECT c.name, COUNT(o.order_id) AS n FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.name HAVING COUNT(o.order_id) > 5",
    "join",
)
add(
    "List the email of every customer who has an active subscription.",
    "SELECT c.email FROM customers c JOIN subscriptions s ON c.customer_id = s.customer_id WHERE s.status = 'active'",
    "join",
)
add(
    "Show the total number of events per customer segment.",
    "SELECT c.segment, COUNT(e.event_id) AS events FROM customers c JOIN events e ON c.customer_id = e.customer_id GROUP BY c.segment",
    "join",
)
add(
    "For each brand, compute total revenue from completed orders.",
    "SELECT p.brand, SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) AS revenue FROM products p JOIN order_items oi ON p.product_id = oi.product_id JOIN orders o ON oi.order_id = o.order_id WHERE o.status = 'completed' GROUP BY p.brand",
    "join",
)
add(
    "List pairs of orders that share the same customer, showing the two order ids.",
    "SELECT o1.order_id AS order_a, o2.order_id AS order_b FROM orders o1 JOIN orders o2 ON o1.customer_id = o2.customer_id AND o1.order_id < o2.order_id LIMIT 50",
    "join",
)
add(
    "Show the product name and brand for every line item in order 250.",
    "SELECT p.name, p.brand FROM order_items oi JOIN products p ON oi.product_id = p.product_id WHERE oi.order_id = 250",
    "join",
)
add(
    "Which customers have subscriptions but no orders? Show their names.",
    "SELECT c.name FROM customers c JOIN subscriptions s ON c.customer_id = s.customer_id LEFT JOIN orders o ON c.customer_id = o.customer_id WHERE o.order_id IS NULL GROUP BY c.name",
    "join",
)
add(
    "Show the number of items and the total amount per order for orders over 500 in total.",
    "SELECT oi.order_id, COUNT(*) AS n_items, SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) AS amount FROM order_items oi GROUP BY oi.order_id HAVING SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) > 500",
    "join",
)
add(
    "List each customer name and their total spend, highest spender first.",
    "SELECT c.name, SUM(o.total_amount) AS spend FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.name ORDER BY spend DESC",
    "join",
)
add(
    "Show the average order value by channel.",
    "SELECT channel, AVG(total_amount) AS avg_value FROM orders GROUP BY channel",
    "groupby_having",
)
add(
    "Which product categories have more than 40 products? Show category and count.",
    "SELECT category, COUNT(*) AS n FROM products GROUP BY category HAVING COUNT(*) > 40",
    "groupby_having",
)
add(
    "List the number of subscriptions per plan.",
    "SELECT plan, COUNT(*) AS n FROM subscriptions GROUP BY plan",
    "groupby_having",
)
add(
    "Show the number of events per device type, most events first.",
    "SELECT device, COUNT(*) AS n FROM events GROUP BY device ORDER BY n DESC",
    "groupby_having",
)

# --------------------------------------------------------------------------- #
# Group by + having
# --------------------------------------------------------------------------- #
add(
    "For each country, count how many customers there are and keep only those with more than 100 customers.",
    "SELECT country, COUNT(*) AS n FROM customers GROUP BY country HAVING COUNT(*) > 100",
    "groupby_having",
)
add(
    "Show order counts per status, only for statuses with at least 500 orders.",
    "SELECT status, COUNT(*) AS n FROM orders GROUP BY status HAVING COUNT(*) >= 500",
    "groupby_having",
)
add(
    "Compute the average product price per category.",
    "SELECT category, AVG(price) AS avg_price FROM products GROUP BY category",
    "groupby_having",
)
add(
    "Find channels with total sales above 10000.",
    "SELECT channel, SUM(total_amount) AS sales FROM orders GROUP BY channel HAVING SUM(total_amount) > 10000",
    "groupby_having",
)
add(
    "List the number of customers per segment, ordered by count descending.",
    "SELECT segment, COUNT(*) AS n FROM customers GROUP BY segment ORDER BY n DESC",
    "groupby_having",
)
add(
    "For each subscription status, compute total MRR and only keep statuses with MRR above 50000.",
    "SELECT status, SUM(mrr) AS total_mrr FROM subscriptions GROUP BY status HAVING SUM(mrr) > 50000",
    "groupby_having",
)
add(
    "Show the average order total per customer segment.",
    "SELECT c.segment, AVG(o.total_amount) AS avg_total FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.segment",
    "groupby_having",
)
add(
    "Count events per day for the last 30 days of the events table.",
    "SELECT event_date, COUNT(*) AS n FROM events GROUP BY event_date ORDER BY event_date DESC LIMIT 30",
    "groupby_having",
)
add(
    "For each product, show the number of orders in which it appears, only products appearing in more than 50 orders.",
    "SELECT p.name, COUNT(*) AS n FROM order_items oi JOIN products p ON oi.product_id = p.product_id GROUP BY p.name HAVING COUNT(*) > 50",
    "groupby_having",
)
add(
    "Compute the total discount given per product category.",
    "SELECT p.category, SUM(oi.discount * oi.quantity * oi.unit_price) AS discount_total FROM order_items oi JOIN products p ON oi.product_id = p.product_id GROUP BY p.category",
    "groupby_having",
)
add(
    "Show the number of orders per year.",
    "SELECT EXTRACT(YEAR FROM order_date) AS yr, COUNT(*) AS n FROM orders GROUP BY yr ORDER BY yr",
    "groupby_having",
)
add(
    "Which brands have an average product price above 250?",
    "SELECT brand, AVG(price) AS avg_price FROM products GROUP BY brand HAVING AVG(price) > 250",
    "groupby_having",
)
add(
    "For each country, show the number of enterprise customers, keeping countries with at least 10.",
    "SELECT country, COUNT(*) AS n FROM customers WHERE segment = 'enterprise' GROUP BY country HAVING COUNT(*) >= 10",
    "groupby_having",
)
add(
    "List the number of items sold per product, top 10.",
    "SELECT p.name, SUM(oi.quantity) AS units FROM products p JOIN order_items oi ON p.product_id = oi.product_id GROUP BY p.name ORDER BY units DESC LIMIT 10",
    "groupby_having",
)
add(
    "Find the average number of events per session.",
    "SELECT AVG(cnt) FROM (SELECT session_id, COUNT(*) AS cnt FROM events GROUP BY session_id) sub",
    "subquery",
)
add(
    "Show the total spend per customer, only customers spending over 5000.",
    "SELECT c.name, SUM(o.total_amount) AS spend FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.name HAVING SUM(o.total_amount) > 5000",
    "groupby_having",
)
add(
    "For each plan, compute the average MRR.",
    "SELECT plan, AVG(mrr) AS avg_mrr FROM subscriptions GROUP BY plan",
    "groupby_having",
)
add(
    "How many distinct customers placed an order each status?",
    "SELECT status, COUNT(DISTINCT customer_id) AS customers FROM orders GROUP BY status",
    "groupby_having",
)

# --------------------------------------------------------------------------- #
# Window functions
# --------------------------------------------------------------------------- #
add(
    "Rank customers by total spend, showing name and rank.",
    "SELECT c.name, SUM(o.total_amount) AS spend, RANK() OVER (ORDER BY SUM(o.total_amount) DESC) AS rnk FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.name",
    "window",
)
add(
    "For each order, show the running total of amount by order date within each channel.",
    "SELECT order_id, channel, total_amount, SUM(total_amount) OVER (PARTITION BY channel ORDER BY order_date, order_id) AS running_total FROM orders",
    "window",
)
add(
    "Number the orders of each customer by date, newest first.",
    "SELECT customer_id, order_id, order_date, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) AS seq FROM orders",
    "window",
)
add(
    "For each product category, show the most expensive product using a window function.",
    "SELECT category, name, price FROM (SELECT category, name, price, ROW_NUMBER() OVER (PARTITION BY category ORDER BY price DESC) AS rn FROM products) t WHERE rn = 1",
    "window",
)
add(
    "Show each customer's total spend and the average spend across all customers as a window.",
    "SELECT c.name, SUM(o.total_amount) AS spend, AVG(SUM(o.total_amount)) OVER () AS avg_spend FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.name",
    "window",
)
add(
    "List each customer's orders with the previous order total using LAG.",
    "SELECT customer_id, order_id, total_amount, LAG(total_amount) OVER (PARTITION BY customer_id ORDER BY order_date, order_id) AS prev_total FROM orders",
    "window",
)
add(
    "Rank products by price within each brand.",
    "SELECT brand, name, price, RANK() OVER (PARTITION BY brand ORDER BY price DESC) AS rnk FROM products",
    "window",
)
add(
    "For each customer, show their largest order total using a window function.",
    "SELECT customer_id, order_id, total_amount FROM (SELECT customer_id, order_id, total_amount, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY total_amount DESC) AS rn FROM orders) t WHERE rn = 1",
    "window",
)
add(
    "Compute the cumulative number of signups by day.",
    "SELECT signup_date, COUNT(*) AS n, SUM(COUNT(*)) OVER (ORDER BY signup_date) AS cumulative FROM customers GROUP BY signup_date",
    "window",
)
add(
    "Show the 3rd highest order total using a window function.",
    "SELECT total_amount FROM (SELECT total_amount, DENSE_RANK() OVER (ORDER BY total_amount DESC) AS rnk FROM orders) t WHERE rnk = 3",
    "window",
)
add(
    "For each event type, show the device with the most events.",
    "SELECT event_type, device, n FROM (SELECT event_type, device, COUNT(*) AS n, ROW_NUMBER() OVER (PARTITION BY event_type ORDER BY COUNT(*) DESC) AS rn FROM events GROUP BY event_type, device) t WHERE rn = 1",
    "window",
)
add(
    "Show each order total along with the median-free percentile rank.",
    "SELECT order_id, total_amount, PERCENT_RANK() OVER (ORDER BY total_amount) AS pr FROM orders",
    "window",
)
add(
    "List customers with more than 10 orders and their rank by order count.",
    "SELECT customer_id, COUNT(*) AS n, RANK() OVER (ORDER BY COUNT(*) DESC) AS rnk FROM orders GROUP BY customer_id HAVING COUNT(*) > 10",
    "window",
)
add(
    "For each month, show revenue and the previous month's revenue.",
    "SELECT m, revenue, LAG(revenue) OVER (ORDER BY m) AS prev_revenue FROM (SELECT DATE_TRUNC('month', order_date) AS m, SUM(total_amount) AS revenue FROM orders GROUP BY 1) t",
    "window",
)
add(
    "Assign each product to a quartile by price.",
    "SELECT name, price, NTILE(4) OVER (ORDER BY price) AS quartile FROM products",
    "window",
)

# --------------------------------------------------------------------------- #
# Date arithmetic
# --------------------------------------------------------------------------- #
add(
    "List orders placed in January 2024.",
    "SELECT order_id FROM orders WHERE order_date >= DATE '2024-01-01' AND order_date < DATE '2024-02-01'",
    "date",
)
add(
    "How many customers signed up in the last 365 days of 2023?",
    "SELECT COUNT(*) FROM customers WHERE signup_date >= DATE '2023-01-01' AND signup_date < DATE '2024-01-01'",
    "date",
)
add(
    "For each order, show the number of days between the customer's signup and the order date.",
    "SELECT o.order_id, DATE_DIFF('day', c.signup_date, o.order_date) AS days_since_signup FROM orders o JOIN customers c ON o.customer_id = c.customer_id",
    "date",
)
add(
    "Show total revenue per month in 2023.",
    "SELECT DATE_TRUNC('month', order_date) AS month, SUM(total_amount) AS revenue FROM orders WHERE order_date >= DATE '2023-01-01' AND order_date < DATE '2024-01-01' GROUP BY month ORDER BY month",
    "date",
)
add(
    "How many events happened on each day of the week?",
    "SELECT DAYNAME(event_date) AS dow, COUNT(*) AS n FROM events GROUP BY dow",
    "date",
)
add(
    "List subscriptions that started more than 180 days after the customer signed up.",
    "SELECT s.subscription_id FROM subscriptions s JOIN customers c ON s.customer_id = c.customer_id WHERE DATE_DIFF('day', c.signup_date, s.start_date) > 180",
    "date",
)
add(
    "Show the number of new customers per month.",
    "SELECT DATE_TRUNC('month', signup_date) AS month, COUNT(*) AS n FROM customers GROUP BY month ORDER BY month",
    "date",
)
add(
    "Find orders placed on weekends.",
    "SELECT order_id FROM orders WHERE DAYOFWEEK(order_date) IN (0, 6)",
    "date",
)
add(
    "Compute the average gap in days between consecutive orders for each customer.",
    "SELECT customer_id, AVG(gap) AS avg_gap FROM (SELECT customer_id, DATE_DIFF('day', LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date), order_date) AS gap FROM orders) t WHERE gap IS NOT NULL GROUP BY customer_id",
    "date",
)
add(
    "List customers who signed up in the same quarter as their first order.",
    "SELECT DISTINCT c.customer_id FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE DATE_TRUNC('quarter', c.signup_date) = DATE_TRUNC('quarter', o.order_date)",
    "date",
)
add(
    "Show events in the 7 days after 2023-06-01.",
    "SELECT event_id FROM events WHERE event_date >= DATE '2023-06-01' AND event_date <= DATE '2023-06-01' + INTERVAL 7 DAY",
    "date",
)
add(
    "For each subscription, compute the duration in days and only keep durations over 100 days.",
    "SELECT subscription_id, DATE_DIFF('day', start_date, COALESCE(end_date, DATE '2024-01-01')) AS duration FROM subscriptions WHERE DATE_DIFF('day', start_date, COALESCE(end_date, DATE '2024-01-01')) > 100",
    "date",
)
add(
    "Show the first and last order date for each customer.",
    "SELECT customer_id, MIN(order_date) AS first_order, MAX(order_date) AS last_order FROM orders GROUP BY customer_id",
    "date",
)
add(
    "Count orders by month name.",
    "SELECT MONTHNAME(order_date) AS month_name, COUNT(*) AS n FROM orders GROUP BY month_name",
    "date",
)
add(
    "List customers whose subscription started within 30 days of signup.",
    "SELECT c.name FROM customers c JOIN subscriptions s ON c.customer_id = s.customer_id WHERE ABS(DATE_DIFF('day', c.signup_date, s.start_date)) <= 30",
    "date",
)

# --------------------------------------------------------------------------- #
# Nested subqueries
# --------------------------------------------------------------------------- #
add(
    "Find products priced above the average product price.",
    "SELECT name, price FROM products WHERE price > (SELECT AVG(price) FROM products)",
    "subquery",
)
add(
    "List customers who spent more than the average customer spend.",
    "SELECT c.name FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.name HAVING SUM(o.total_amount) > (SELECT AVG(total) FROM (SELECT SUM(total_amount) AS total FROM orders GROUP BY customer_id) t)",
    "subquery",
)
add(
    "Find the customer with the single highest order total.",
    "SELECT c.name FROM customers c WHERE c.customer_id = (SELECT customer_id FROM orders ORDER BY total_amount DESC LIMIT 1)",
    "subquery",
)
add(
    "List products that have never been ordered.",
    "SELECT name FROM products WHERE product_id NOT IN (SELECT DISTINCT product_id FROM order_items)",
    "subquery",
)
add(
    "Find customers who placed at least one order above 1000.",
    "SELECT DISTINCT name FROM customers WHERE customer_id IN (SELECT customer_id FROM orders WHERE total_amount > 1000)",
    "subquery",
)
add(
    "Show the category with the highest average price.",
    "SELECT category FROM products GROUP BY category ORDER BY AVG(price) DESC LIMIT 1",
    "subquery",
)
add(
    "Find orders whose total is greater than every order from the 'partner' channel.",
    "SELECT order_id FROM orders WHERE total_amount > ALL (SELECT total_amount FROM orders WHERE channel = 'partner')",
    "subquery",
)
add(
    "List customers whose total spend is above the 90th percentile of customer spend.",
    "SELECT c.name FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.name HAVING SUM(o.total_amount) > (SELECT QUANTILE_CONT(total, 0.9) FROM (SELECT SUM(total_amount) AS total FROM orders GROUP BY customer_id) t)",
    "subquery",
)
add(
    "Find the most popular product by number of orders.",
    "SELECT p.name FROM products p JOIN order_items oi ON p.product_id = oi.product_id GROUP BY p.name ORDER BY COUNT(*) DESC LIMIT 1",
    "subquery",
)
add(
    "Show countries that have more than the average number of customers per country.",
    "SELECT country FROM customers GROUP BY country HAVING COUNT(*) > (SELECT AVG(c) FROM (SELECT COUNT(*) AS c FROM customers GROUP BY country) t)",
    "subquery",
)
add(
    "List orders that contain a product from the 'toys' category.",
    "SELECT DISTINCT o.order_id FROM orders o WHERE o.order_id IN (SELECT oi.order_id FROM order_items oi JOIN products p ON oi.product_id = p.product_id WHERE p.category = 'toys')",
    "subquery",
)
add(
    "Find customers who have exactly one subscription.",
    "SELECT customer_id FROM subscriptions GROUP BY customer_id HAVING COUNT(*) = 1",
    "subquery",
)
add(
    "Show the second most expensive product.",
    "SELECT name FROM products ORDER BY price DESC LIMIT 1 OFFSET 1",
    "subquery",
)
add(
    "Find brands whose average price is above the overall average price.",
    "SELECT brand FROM products GROUP BY brand HAVING AVG(price) > (SELECT AVG(price) FROM products)",
    "subquery",
)
add(
    "List the customers with the most events.",
    "SELECT c.name FROM customers c JOIN events e ON c.customer_id = e.customer_id GROUP BY c.name ORDER BY COUNT(*) DESC LIMIT 1",
    "subquery",
)

# --------------------------------------------------------------------------- #
# Ambiguous / requires qualification
# --------------------------------------------------------------------------- #
add(
    "Join orders and order_items and select the order date and quantity, qualifying all columns.",
    "SELECT o.order_date, oi.quantity FROM orders o JOIN order_items oi ON o.order_id = oi.order_id",
    "ambiguous",
)
add(
    "List the customer id and product id that appear in the most order lines.",
    "SELECT o.customer_id, oi.product_id FROM orders o JOIN order_items oi ON o.order_id = oi.order_id GROUP BY 1, 2 ORDER BY COUNT(*) DESC LIMIT 1",
    "ambiguous",
)
add(
    "Show product names that appear in orders placed by US customers.",
    "SELECT DISTINCT p.name FROM products p JOIN order_items oi ON p.product_id = oi.product_id JOIN orders o ON oi.order_id = o.order_id JOIN customers c ON o.customer_id = c.customer_id WHERE c.country = 'US'",
    "ambiguous",
)
add(
    "For each country, find the number of distinct products ordered and the total revenue.",
    "SELECT c.country, COUNT(DISTINCT oi.product_id) AS n_products, SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) AS revenue FROM customers c JOIN orders o ON c.customer_id = o.customer_id JOIN order_items oi ON o.order_id = oi.order_id GROUP BY c.country",
    "ambiguous",
)
add(
    "List the name of each product and the name of the customer who bought it, for order 42.",
    "SELECT p.name AS product_name, c.name AS customer_name FROM order_items oi JOIN products p ON oi.product_id = p.product_id JOIN orders o ON oi.order_id = o.order_id JOIN customers c ON o.customer_id = c.customer_id WHERE oi.order_id = 42",
    "ambiguous",
)
add(
    "Write a query joining events and customers selecting the event date and customer country.",
    "SELECT e.event_date, c.country FROM events e JOIN customers c ON e.customer_id = c.customer_id",
    "ambiguous",
)
add(
    "Find the top product by units within each category, returning the category and product name.",
    "SELECT category, name FROM (SELECT p.category, p.name, SUM(oi.quantity) AS units, ROW_NUMBER() OVER (PARTITION BY p.category ORDER BY SUM(oi.quantity) DESC) AS rn FROM products p JOIN order_items oi ON p.product_id = oi.product_id GROUP BY p.category, p.name) t WHERE rn = 1",
    "ambiguous",
)
add(
    "Compute revenue by customer segment and channel in one query, qualifying every column.",
    "SELECT c.segment, o.channel, SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) AS revenue FROM customers c JOIN orders o ON c.customer_id = o.customer_id JOIN order_items oi ON o.order_id = oi.order_id GROUP BY c.segment, o.channel",
    "ambiguous",
)
add(
    "Show each subscription's plan and the customer's country, qualifying the join columns.",
    "SELECT s.plan, c.country FROM subscriptions s JOIN customers c ON s.customer_id = c.customer_id",
    "ambiguous",
)
add(
    "List customer names who ordered a product priced above 300.",
    "SELECT DISTINCT c.name FROM customers c JOIN orders o ON c.customer_id = o.customer_id JOIN order_items oi ON o.order_id = oi.order_id JOIN products p ON oi.product_id = p.product_id WHERE p.price > 300",
    "ambiguous",
)
add(
    "For each order, show the order id, customer name and number of items.",
    "SELECT o.order_id, c.name, COUNT(oi.item_id) AS n_items FROM orders o JOIN customers c ON o.customer_id = c.customer_id JOIN order_items oi ON o.order_id = oi.order_id GROUP BY o.order_id, c.name",
    "ambiguous",
)
add(
    "Find the total quantity of products sold per country.",
    "SELECT c.country, SUM(oi.quantity) AS units FROM customers c JOIN orders o ON c.customer_id = o.customer_id JOIN order_items oi ON o.order_id = oi.order_id GROUP BY c.country",
    "ambiguous",
)
add(
    "List each brand's number of products and total units sold.",
    "SELECT p.brand, COUNT(DISTINCT p.product_id) AS n_products, SUM(oi.quantity) AS units FROM products p LEFT JOIN order_items oi ON p.product_id = oi.product_id GROUP BY p.brand",
    "ambiguous",
)
add(
    "Show each customer's name and count of distinct event types.",
    "SELECT c.name, COUNT(DISTINCT e.event_type) AS n_types FROM customers c JOIN events e ON c.customer_id = e.customer_id GROUP BY c.name",
    "ambiguous",
)
add(
    "Join subscriptions and orders to show each customer's plan and their number of orders.",
    "SELECT c.customer_id, s.plan, COUNT(DISTINCT o.order_id) AS n_orders FROM customers c JOIN subscriptions s ON c.customer_id = s.customer_id LEFT JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.customer_id, s.plan",
    "ambiguous",
)
add(
    "What percentage of orders are cancelled?",
    "SELECT 100.0 * SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) / COUNT(*) AS pct FROM orders",
    "aggregate",
)
add(
    "Show the distribution of customers across countries as a share of the total.",
    "SELECT country, COUNT(*) * 1.0 / (SELECT COUNT(*) FROM customers) AS share FROM customers GROUP BY country",
    "subquery",
)
add(
    "For each device, compute the conversion rate from checkout_start to purchase events.",
    "SELECT device, SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) * 1.0 / NULLIF(SUM(CASE WHEN event_type = 'checkout_start' THEN 1 ELSE 0 END), 0) AS conversion FROM events GROUP BY device",
    "aggregate",
)
add(
    "Compute the gross margin per product category using price and cost.",
    "SELECT p.category, SUM(oi.quantity * (p.price - p.cost)) AS margin FROM order_items oi JOIN products p ON oi.product_id = p.product_id GROUP BY p.category",
    "join",
)
add(
    "List products whose price is above their brand's average price.",
    "SELECT name FROM products p WHERE price > (SELECT AVG(price) FROM products p2 WHERE p2.brand = p.brand)",
    "subquery",
)
add(
    "Show the number of orders and total revenue per quarter.",
    "SELECT DATE_TRUNC('quarter', order_date) AS quarter, COUNT(*) AS n, SUM(total_amount) AS revenue FROM orders GROUP BY quarter ORDER BY quarter",
    "date",
)
add(
    "Find the customer segment with the highest average subscription MRR.",
    "SELECT c.segment, AVG(s.mrr) AS avg_mrr FROM customers c JOIN subscriptions s ON c.customer_id = s.customer_id GROUP BY c.segment ORDER BY avg_mrr DESC LIMIT 1",
    "join",
)
add(
    "List each country's number of active subscriptions.",
    "SELECT c.country, COUNT(*) AS n FROM customers c JOIN subscriptions s ON c.customer_id = s.customer_id WHERE s.status = 'active' GROUP BY c.country",
    "join",
)
add(
    "Show total revenue for orders that include more than 3 line items.",
    "SELECT SUM(o.total_amount) FROM orders o WHERE o.order_id IN (SELECT order_id FROM order_items GROUP BY order_id HAVING COUNT(*) > 3)",
    "subquery",
)
add(
    "For each customer, show the fraction of their orders that were refunded.",
    "SELECT customer_id, SUM(CASE WHEN status = 'refunded' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS refund_rate FROM orders GROUP BY customer_id",
    "aggregate",
)
add(
    "Find the product with the highest margin per unit sold.",
    "SELECT p.name FROM products p JOIN order_items oi ON p.product_id = oi.product_id GROUP BY p.name ORDER BY SUM(oi.quantity * (p.price - p.cost)) DESC LIMIT 1",
    "join",
)
add(
    "List the top 3 countries by number of enterprise customers.",
    "SELECT country, COUNT(*) AS n FROM customers WHERE segment = 'enterprise' GROUP BY country ORDER BY n DESC LIMIT 3",
    "groupby_having",
)
add("Compute the median order total.", "SELECT MEDIAN(total_amount) FROM orders", "aggregate")
add(
    "Show the number of customers who have both an active subscription and at least one order.",
    "SELECT COUNT(DISTINCT c.customer_id) FROM customers c JOIN subscriptions s ON c.customer_id = s.customer_id JOIN orders o ON c.customer_id = o.customer_id WHERE s.status = 'active'",
    "join",
)


def main() -> None:
    os.makedirs("data", exist_ok=True)
    out = []
    for i, task in enumerate(T, start=1):
        row = {"task_id": f"t{i:04d}", **task}
        out.append(row)
    path = os.path.join("data", "tasks.jsonl")
    with open(path, "w") as fh:
        for row in out:
            fh.write(json.dumps(row) + "\n")
    cats: dict[str, int] = {}
    for row in out:
        cats[row["category"]] = cats.get(row["category"], 0) + 1
    print(f"wrote {len(out)} tasks -> {path}")
    print("categories:", cats)


if __name__ == "__main__":
    main()
