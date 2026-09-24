import { useAuth } from '../lib/AuthContext';
import { TAGLINES } from '../lib/taglines';

export default function PatientDashboard() {
  const { profile } = useAuth();

  return (
    <div className="min-h-screen bg-cardio-cream p-8">
      <h1 className="text-2xl font-semibold text-cardio-teal">
        Welcome, {profile?.full_name || 'Patient'}
      </h1>
      <p className="text-cardio-ink/70 italic mt-1">{TAGLINES[1]}</p>

      <div className="grid md:grid-cols-2 gap-6 mt-8">
        <section className="bg-white rounded-xl shadow-sm p-6">
          <h2 className="font-medium text-cardio-ink mb-2">My Cases</h2>
          <p className="text-sm text-cardio-ink/60">No cases yet.</p>
        </section>
        <section className="bg-white rounded-xl shadow-sm p-6">
          <h2 className="font-medium text-cardio-ink mb-2">Attending Doctor</h2>
          <p className="text-sm text-cardio-ink/60">Not assigned yet.</p>
        </section>
      </div>
    </div>
  );
}
