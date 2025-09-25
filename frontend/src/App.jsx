import { useState } from 'react';

function App() {
  const [query, setQuery] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [model, setModel] = useState('mpnet');

  const modelOptions = [
    { value: 'mpnet', label: 'MPNet' },
    { value: 'bert', label: 'BERT' },
    { value: 'xlnet', label: 'XLNet' },
    { value: 'albert', label: 'ALBERT' },
    { value: 'distilbert', label: 'DistilBERT' },
  ];

  const handleSearch = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`http://localhost:8000/recommend?topic=${encodeURIComponent(query)}&model=${encodeURIComponent(model)}`);
      if (!response.ok) throw new Error('Failed to fetch recommendations');
      const data = await response.json();
      setRecommendations(data.results || []);
    } catch (err) {
      setError(err.message);
      setRecommendations([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ fontFamily: 'Arial, sans-serif', padding: '2rem', maxWidth: '600px', margin: '0 auto' }}>
      <header style={{ marginBottom: '2rem' }}>
        <h1>Researcher Recommender</h1>
        <p>Find and discover researchers based on your interests.</p>
      </header>
      <section style={{ marginBottom: '2rem' }}>
        <form onSubmit={handleSearch}>
          <label htmlFor="model-select" style={{ fontWeight: 'bold', marginBottom: '0.5rem', display: 'block' }}>Select Transformer Model:</label>
          <select
            id="model-select"
            value={model}
            onChange={e => setModel(e.target.value)}
            style={{ width: '100%', padding: '0.5rem', fontSize: '1rem', marginBottom: '1rem' }}
          >
            {modelOptions.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
          <input
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Search for a researcher..."
            style={{ width: '100%', padding: '0.5rem', fontSize: '1rem' }}
          />
          <button type="submit" style={{ marginTop: '1rem', padding: '0.5rem 1rem', fontSize: '1rem' }}>Search</button>
        </form>
      </section>
      <section>
        <h2>Recommendations</h2>
  <div style={{ background: '#222', padding: '1rem', borderRadius: '8px' }}>
          {loading && <p>Loading...</p>}
          {error && <p style={{ color: 'red' }}>{error}</p>}
          {!loading && !error && recommendations.length === 0 && <p>No recommendations yet.</p>}
          <ul style={{ listStyle: 'none', padding: 0 }}>
            {recommendations.map((rec, idx) => (
              <li key={idx} style={{ marginBottom: '1rem', background: '#fff', color: '#222', padding: '1rem', borderRadius: '8px', boxShadow: '0 1px 4px rgba(0,0,0,0.10)' }}>
                <strong style={{ color: '#1a237e' }}>{rec.researcher}</strong> <br />
                <span><b>Field:</b> {rec.field}</span> <br />
                <span><b>Score:</b> {rec.score}</span> <br />
                <span><b>Top Topics:</b> {rec.top_topics && rec.top_topics.join(', ')}</span>
              </li>
            ))}
          </ul>
        </div>
      </section>
    </div>
  );
}

export default App