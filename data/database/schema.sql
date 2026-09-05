-- Reference relational schema for production ingestion. Customer IDs join all operational entities.
CREATE TABLE customers (customer_id TEXT PRIMARY KEY);
CREATE TABLE invoices (invoice_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers(customer_id));
CREATE TABLE payments (payment_id TEXT PRIMARY KEY, invoice_id TEXT NOT NULL REFERENCES invoices(invoice_id), customer_id TEXT NOT NULL REFERENCES customers(customer_id));
CREATE TABLE transactions (transaction_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers(customer_id));
CREATE TABLE subscriptions (subscription_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers(customer_id));
CREATE TABLE customer_interactions (interaction_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers(customer_id));
CREATE TABLE recovery_actions (action_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers(customer_id));
CREATE TABLE predictions (customer_id TEXT PRIMARY KEY REFERENCES customers(customer_id));
