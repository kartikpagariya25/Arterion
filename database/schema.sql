-- CardioSense core schema (Supabase Postgres)

create type user_role as enum ('patient', 'doctor');
create type case_type as enum ('vessel_analysis', 'ef_estimation', 'lv_outline', 'dominance', 'arrhythmia');
create type case_status as enum ('pending', 'processing', 'completed', 'failed');

create table users (
  id uuid primary key references auth.users(id) on delete cascade,
  role user_role not null,
  full_name text not null,
  email text not null unique,
  created_at timestamptz not null default now()
);

create table doctors (
  id uuid primary key references users(id) on delete cascade,
  specialization text,
  registration_no text,
  phone text
);

create table patients (
  id uuid primary key references users(id) on delete cascade,
  assigned_doctor_id uuid references doctors(id),
  dob date,
  gender text,
  blood_group text,
  phone text,
  address text,
  medical_history jsonb default '[]',
  medications jsonb default '[]',
  allergies jsonb default '[]',
  lifestyle jsonb default '{}',
  created_at timestamptz not null default now()
);

create table vitals (
  id uuid primary key default gen_random_uuid(),
  patient_id uuid references patients(id) on delete cascade,
  vital_type text not null,
  value numeric not null,
  unit text,
  recorded_at timestamptz not null default now()
);

create table cases (
  id uuid primary key default gen_random_uuid(),
  patient_id uuid references patients(id) on delete cascade,
  doctor_id uuid references doctors(id),
  case_type case_type not null,
  status case_status not null default 'pending',
  notes text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table case_files (
  id uuid primary key default gen_random_uuid(),
  case_id uuid references cases(id) on delete cascade,
  file_url text not null,
  file_type text not null,
  uploaded_at timestamptz not null default now()
);

create table case_results (
  id uuid primary key default gen_random_uuid(),
  case_id uuid references cases(id) on delete cascade,
  result_json jsonb not null,
  result_summary text,
  confidence numeric,
  reviewed_by_doctor boolean default false,
  created_at timestamptz not null default now()
);

create table audit_logs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references users(id),
  action text not null,
  entity text not null,
  entity_id uuid,
  timestamp timestamptz not null default now()
);

-- Row Level Security
alter table users enable row level security;
alter table doctors enable row level security;
alter table patients enable row level security;
alter table cases enable row level security;
alter table case_files enable row level security;
alter table case_results enable row level security;
alter table vitals enable row level security;

create policy "users read own row" on users for select using (auth.uid() = id);

create policy "doctors read all doctors" on doctors for select using (
  exists (select 1 from users where id = auth.uid() and role = 'doctor')
);

create policy "doctors read all patients" on patients for select using (
  exists (select 1 from users where id = auth.uid() and role = 'doctor')
);
create policy "patients read own row" on patients for select using (auth.uid() = id);
create policy "doctors update patients" on patients for update using (
  exists (select 1 from users where id = auth.uid() and role = 'doctor')
);

create policy "doctors read all cases" on cases for select using (
  exists (select 1 from users where id = auth.uid() and role = 'doctor')
);
create policy "patients read own cases" on cases for select using (patient_id = auth.uid());
create policy "doctors manage cases" on cases for all using (
  exists (select 1 from users where id = auth.uid() and role = 'doctor')
);
