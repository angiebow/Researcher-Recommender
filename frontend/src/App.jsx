import './App.css';

function App() {
  return (
    <div style={{ fontFamily: 'Arial, sans-serif', padding: '2rem', maxWidth: '600px', margin: '0 auto' }}>
      <header style={{ marginBottom: '2rem' }}>
        <h1>Researcher Recommender</h1>
        <p>Find and discover researchers based on your interests.</p>
      </header>
      <section style={{ marginBottom: '2rem' }}>
        <input
          type="text"
          placeholder="Search for a researcher..."
          style={{ width: '100%', padding: '0.5rem', fontSize: '1rem' }}
        />
      </section>
      <section>
        <h2>Recommendations</h2>
        <div style={{ background: '#f4f4f4', padding: '1rem', borderRadius: '8px' }}>
          <p>No recommendations yet.</p>
        </div>
      </section>
    </div>
  );
}

export default App
