import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Simple dataset
data = [
    ("Win a free iPhone! Click now!", 1),
    ("We have a meeting tomorrow, don’t forget", 0), 
    ("HURRY UP! 50% discount!", 1),
    ("Shall we go out for coffee?", 0),
    ("FREE MONEY! Click the link!", 1),
    ("Shopping list: milk, bread", 0),
    ("Become a millionaire! Click!", 1),
    ("Do you have a movie recommendation?", 0),
    ("WIN WIN! Money!", 1),
    ("Your doctor appointment is tomorrow", 0),
    ("Your credit card is ready! Take it!", 1),
    ("How’s the project going?", 0),
]

class SpamModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.5),  # Higher dropout
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

# Prepare data
texts = [d[0] for d in data]
labels = [d[1] for d in data]

# TF-IDF (limit features)
vectorizer = TfidfVectorizer(max_features=100, stop_words=None)
X = vectorizer.fit_transform(texts).toarray()
y = np.array(labels, dtype=np.float32)
print(X.shape)

# Model and training
model = SpamModel(X.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Higher lr

# Training (few epochs)
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(torch.FloatTensor(X))
    loss = criterion(outputs.squeeze(), torch.FloatTensor(y))
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.3f}')

# Test function
def test_spam(text):
    model.eval()
    with torch.no_grad():
        vec = vectorizer.transform([text]).toarray()
        spam_prob = model(torch.FloatTensor(vec)).item()
        
        if spam_prob > 0.5:
            result = "SPAM"
            confidence = spam_prob
        else:
            result = "HAM" 
            confidence = 1 - spam_prob  # HAM probability
            
        return f"'{text}' -> {result} ({confidence:.3f})"

# Test samples
test_texts = [
    "Win a free gift now! Click!",
    "Are you coming to the meeting tomorrow?", 
    "HURRY! Limited time offer!",
    "Shall we have coffee?",
    "Would you like to drink free coffee?",
]

print("\nTEST RESULTS:")
for text in test_texts:
    print(test_spam(text))
