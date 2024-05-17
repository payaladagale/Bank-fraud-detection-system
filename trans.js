// script.js
document.getElementById('transaction-form').addEventListener('submit', function(event) {
  event.preventDefault();
  const amount = document.getElementById('amount').value;
  const date = document.getElementById('date').value;
  const merchant = document.getElementById('merchant').value;
  const cardNumber = document.getElementById('card-number').value;
  
  // Simulate fraud detection (dummy logic)
  const isFraudulent = Math.random() < 0.1; // 10% chance of being fraudulent
  
  const transaction = {
    amount,
    date,
    merchant,
    cardNumber,
    isFraudulent
  };

  displayTransaction(transaction);
  if (isFraudulent) {
    displayAlert(transaction);
  }
});

function displayTransaction(transaction) {
  const transactionList = document.getElementById('transactions');
  const listItem = document.createElement('li');
  listItem.textContent = `Amount: $${transaction.amount}, Date: ${transaction.date}, Merchant: ${transaction.merchant}, Card Number: ${transaction.cardNumber}`;
  transactionList.appendChild(listItem);
}

function displayAlert(transaction) {
  const alertsDiv = document.getElementById('alerts');
  const alertDiv = document.createElement('div');
  alertDiv.classList.add('alert');
  alertDiv.textContent = `Fraudulent transaction detected! Amount: $${transaction.amount}, Date: ${transaction.date}, Merchant: ${transaction.merchant}, Card Number: ${transaction.cardNumber}`;
  alertsDiv.appendChild(alertDiv);
}
