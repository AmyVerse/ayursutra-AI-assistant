# Next.js Integration Examples

## API Routes for Next.js

### pages/api/chat.js

```javascript
export default async function handler(req, res) {
  if (req.method === "POST") {
    try {
      const response = await fetch(`${process.env.AYURSUTRA_API_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: req.body.message,
          user_id: req.body.user_id || "anonymous",
        }),
      });

      const data = await response.json();
      res.status(200).json(data);
    } catch (error) {
      res.status(500).json({ error: "Failed to get AI response" });
    }
  }
}
```

### pages/api/medicines/search.js

```javascript
export default async function handler(req, res) {
  if (req.method === "POST") {
    try {
      const response = await fetch(
        `${process.env.AYURSUTRA_API_URL}/medicines/search`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            medicine_name: req.body.medicine_name,
            max_results: req.body.max_results || 10,
          }),
        }
      );

      const data = await response.json();
      res.status(200).json(data);
    } catch (error) {
      res.status(500).json({ error: "Failed to search medicines" });
    }
  }
}
```

## React Components

### components/ChatBot.jsx

```jsx
import { useState } from "react";

export default function ChatBot() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: input }),
      });

      const data = await response.json();

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.response,
          suggested_medicines: data.suggested_medicines,
        },
      ]);
    } catch (error) {
      console.error("Chat error:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <p>{msg.content}</p>
            {msg.suggested_medicines && (
              <div className="suggested-medicines">
                <h4>Suggested Medicines:</h4>
                <ul>
                  {msg.suggested_medicines.map((med, i) => (
                    <li key={i}>{med}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="input-area">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about your health concerns..."
          onKeyPress={(e) => e.key === "Enter" && sendMessage()}
        />
        <button onClick={sendMessage} disabled={loading}>
          {loading ? "Sending..." : "Send"}
        </button>
      </div>
    </div>
  );
}
```

### components/MedicineSearch.jsx

```jsx
import { useState } from "react";

export default function MedicineSearch() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const searchMedicines = async () => {
    if (!query.trim()) return;

    setLoading(true);
    try {
      const response = await fetch("/api/medicines/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ medicine_name: query, max_results: 10 }),
      });

      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error("Search error:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="medicine-search">
      <div className="search-input">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search for medicines..."
          onKeyPress={(e) => e.key === "Enter" && searchMedicines()}
        />
        <button onClick={searchMedicines} disabled={loading}>
          {loading ? "Searching..." : "Search"}
        </button>
      </div>

      {results && (
        <div className="results">
          <h3>Real-time Results ({results.realtime_results.length})</h3>
          {results.realtime_results.map((med, idx) => (
            <div key={idx} className="medicine-card">
              <h4>{med.Medicine}</h4>
              <p>Price: {med.Price}</p>
              <p>Source: {med.Source}</p>
              <p>Stock: {med.Stock}</p>
              <a href={med.Link} target="_blank" rel="noopener noreferrer">
                View Product
              </a>
            </div>
          ))}

          <h3>Local Database Results ({results.local_results.length})</h3>
          {results.local_results.map((med, idx) => (
            <div key={idx} className="medicine-card">
              <h4>{med.Medicine}</h4>
              <p>Brand: {med.Brand}</p>
              <p>Price: {med.Price}</p>
              <p>Rating: {med.Rating}</p>
              <a href={med.Link} target="_blank" rel="noopener noreferrer">
                Buy Now
              </a>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```

## Environment Variables (.env.local)

```
AYURSUTRA_API_URL=http://localhost:8000
# For production:
# AYURSUTRA_API_URL=https://your-railway-app.railway.app
```
