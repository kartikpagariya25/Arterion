import { useState } from 'react';
import { useAuth } from '../lib/AuthContext';

export default function DoctorDashboard() {
  const { profile } = useAuth();
  const [tab, setTab] = useState('mine');

  return (
    <div className="min-h-screen bg-cardio-ink p-8 text-white">
      <h1 className="text-2xl font-semibold">Dr. {profile?.full_name || ''}</h1>

      <div className="flex gap-4 mt-6 border-b border-white/10">
        <button
          onClick={() => setTab('mine')}
          className={`pb-2 ${tab === 'mine' ? 'border-b-2 border-cardio-coral' : 'text-white/50'}`}
        >
          My Patients
        </button>
        <button
          onClick={() => setTab('all')}
          className={`pb-2 ${tab === 'all' ? 'border-b-2 border-cardio-coral' : 'text-white/50'}`}
        >
          All Patients
        </button>
      </div>

      <div className="mt-6 bg-white/5 rounded-xl p-6">
        <p className="text-white/60 text-sm">
          {tab === 'mine' ? 'Assigned patients will appear here.' : 'Full patient directory will appear here.'}
        </p>
      </div>
    </div>
  );
}
