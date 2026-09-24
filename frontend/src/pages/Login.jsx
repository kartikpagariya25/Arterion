import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { supabase } from '../lib/supabaseClient';
import { TAGLINES } from '../lib/taglines';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  async function handleLogin(e) {
    e.preventDefault();
    setError('');
    const { data, error } = await supabase.auth.signInWithPassword({ email, password });
    if (error) return setError(error.message);

    const { data: userRow } = await supabase
      .from('users')
      .select('role')
      .eq('id', data.user.id)
      .single();

    navigate(userRow?.role === 'doctor' ? '/doctor' : '/patient');
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-cardio-cream px-4">
      <div className="w-full max-w-md">
        <h1 className="text-3xl font-semibold text-cardio-teal mb-1">CardioSense</h1>
        <p className="text-sm text-cardio-ink/70 mb-6">AI That Understands the Heart</p>

        <form onSubmit={handleLogin} className="bg-white rounded-xl shadow-sm p-6 space-y-4">
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full border rounded-lg px-3 py-2"
            required
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full border rounded-lg px-3 py-2"
            required
          />
          {error && <p className="text-cardio-coral text-sm">{error}</p>}
          <button
            type="submit"
            className="w-full bg-cardio-teal text-white rounded-lg py-2 font-medium"
          >
            Log In
          </button>
        </form>

        <p className="text-center text-sm text-cardio-ink/60 mt-6 italic">
          {TAGLINES[Math.floor(Math.random() * TAGLINES.length)]}
        </p>
      </div>
    </div>
  );
}
