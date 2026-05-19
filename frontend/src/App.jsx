import { useEffect, useMemo, useState, useCallback } from 'react'
import { AreaChart, Area, BarChart, Bar, LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts'

const API = 'http://localhost:8000'

// ─── Hook fetch générique ─────────────────────────────────────────────────────
function useApi(endpoint) {
  const [data, setData]       = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState(null)

  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API}${endpoint}`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setData(await res.json())
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [endpoint])

  useEffect(() => { fetchData() }, [fetchData])
  return { data, loading, error, refetch: fetchData }
}

// ─── Données statiques (fallback / UI non connectée) ──────────────────────────
const sections = [
  { id: 'dashboard',    label: 'Dashboard',     group: 'Navigation' },
  { id: 'resultats',    label: 'Résultats',      group: 'Navigation' },
  { id: 'comparaison',  label: 'Comparaison',    group: 'Navigation' },
  { id: 'prediction',   label: 'Prédiction',     group: 'Navigation' },
  { id: 'historique',   label: 'Historique',     group: 'MLOps' },
  { id: 'versions',     label: 'Versions',       group: 'MLOps' },
  { id: 'drift',        label: 'Drift Monitor',  group: 'MLOps' },
]

// ─── Composants utilitaires ───────────────────────────────────────────────────
function Spinner() {
  return <div className="flex items-center justify-center py-12 text-slate-400 text-sm">Chargement...</div>
}

function ErrorBox({ message }) {
  return <div className="rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-600">Erreur : {message}</div>
}

function StatCard({ label, value, sub }) {
  return (
    <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
      <p className="text-sm text-slate-500">{label}</p>
      <p className="mt-3 text-3xl font-semibold text-slate-900">{value ?? '—'}</p>
      {sub && <p className="mt-1 text-xs text-slate-400">{sub}</p>}
    </div>
  )
}

// ─── App principale ───────────────────────────────────────────────────────────
export default function App() {
  const [activeSection, setActiveSection] = useState('dashboard')
  const [notifications, setNotifications] = useState([{ id: 1, text: 'Interface chargée', type: 'success' }])

  // Données backend
  const { data: runsData,    loading: runsLoading,    error: runsError    } = useApi('/api/runs')
  const { data: bestRun,     loading: bestLoading,    error: bestError    } = useApi('/api/best-run')
  const { data: registryData,loading: regLoading,     error: regError     } = useApi('/api/registry')
  const { data: driftData,   loading: driftLoading,   error: driftError   } = useApi('/api/drift')

  // État formulaire prédiction
  const [form, setForm] = useState({
    Compacite: 0.98, Surface_Totale: 514.5, Surface_Murs: 294.0,
    Surface_Toit: 110.25, Hauteur: 7.0, Orientation: 2,
    Surface_Vitree: 0.0, Distribution_Vitree: 0,
  })
  const [prediction, setPrediction] = useState(null)
  const [predLoading, setPredLoading] = useState(false)

  useEffect(() => {
    if (!notifications.length) return
    const t = setTimeout(() => setNotifications(p => p.slice(1)), 3500)
    return () => clearTimeout(t)
  }, [notifications])

  const notify = (text, type = 'info') =>
    setNotifications(p => [...p, { id: Date.now(), text, type }])

  const handlePredict = async () => {
    setPredLoading(true)
    try {
      const res = await fetch(`${API}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      const json = await res.json()
      setPrediction(json.prediction)
      notify(`Prédiction : ${json.prediction} ${json.unit}`, 'success')
    } catch {
      notify('Erreur lors de la prédiction.', 'error')
    } finally {
      setPredLoading(false)
    }
  }

  // Données pour graphiques
  const runs = runsData?.runs ?? []

  const comparisonChartData = useMemo(() =>
    runs.slice(0, 8).map(r => ({
      name:  r.model.length > 14 ? r.model.slice(0, 14) + '…' : r.model,
      r2:    r.r2,
      rmse:  r.rmse,
    })),
    [runs]
  )

  const trendData = useMemo(() =>
    runs.slice(0, 10).reverse().map((r, i) => ({
      name:  `Run ${i + 1}`,
      score: Math.round(r.r2 * 100),
    })),
    [runs]
  )

  // ─── Sections ───────────────────────────────────────────────────────────────
  const renderSection = () => {
    switch (activeSection) {

      // ── Dashboard ──────────────────────────────────────────────────────────
      case 'dashboard':
        return (
          <div className="space-y-6">
            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <h1 className="text-2xl font-semibold text-slate-900">Tableau de bord</h1>
              <p className="mt-1 text-sm text-slate-500">Vue globale — données en temps réel depuis MLflow.</p>
            </div>

            {bestLoading ? <Spinner /> : bestError ? <ErrorBox message={bestError} /> : (
              <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-4">
                <StatCard label="Meilleur modèle"     value={bestRun?.model}               />
                <StatCard label="R² (meilleur run)"   value={bestRun?.r2}                  />
                <StatCard label="MAE"                 value={bestRun?.mae}                 />
                <StatCard label="RMSE"                value={bestRun?.rmse}                />
              </div>
            )}

            <div className="grid gap-6 lg:grid-cols-[1.3fr_0.7fr]">
              <div className="rounded-3xl border border-slate-200 bg-slate-50 p-6 shadow-sm">
                <h2 className="text-base font-semibold text-slate-900">Évolution du R² par run</h2>
                {runsLoading ? <Spinner /> : (
                  <div className="mt-5 h-56">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={trendData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                        <defs>
                          <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#10b981" stopOpacity={0.35} />
                            <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" />
                        <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#64748b' }} />
                        <YAxis tick={{ fontSize: 11, fill: '#64748b' }} />
                        <Tooltip formatter={v => `${v}%`} />
                        <Area type="monotone" dataKey="score" stroke="#10b981" fill="url(#colorScore)" strokeWidth={3} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>

              <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
                <h2 className="text-base font-semibold text-slate-900">Actions rapides</h2>
                <div className="mt-5 space-y-3">
                  <button onClick={() => setActiveSection('prediction')} className="w-full rounded-2xl bg-slate-900 px-4 py-3 text-sm font-semibold text-white hover:bg-slate-800">
                    Faire une prédiction
                  </button>
                  <button onClick={() => setActiveSection('historique')} className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700 hover:bg-slate-100">
                    Voir l'historique des runs
                  </button>
                  <button onClick={() => setActiveSection('drift')} className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700 hover:bg-slate-100">
                    Monitorer le drift
                  </button>
                </div>
              </div>
            </div>
          </div>
        )

      // ── Résultats ──────────────────────────────────────────────────────────
      case 'resultats':
        return (
          <div className="space-y-6">
            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <h2 className="text-xl font-semibold text-slate-900">Résultats — Meilleur modèle</h2>
            </div>
            {bestLoading ? <Spinner /> : bestError ? <ErrorBox message={bestError} /> : (
              <>
                <div className="grid gap-6 sm:grid-cols-3">
                  <StatCard label="R²"   value={bestRun?.r2}   sub="Coefficient de détermination" />
                  <StatCard label="MAE"  value={bestRun?.mae}  sub="Mean Absolute Error" />
                  <StatCard label="RMSE" value={bestRun?.rmse} sub="Root Mean Squared Error" />
                </div>
                <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
                  <h3 className="text-sm font-semibold text-slate-900 mb-4">Paramètres du meilleur run</h3>
                  <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                    {Object.entries(bestRun?.params ?? {}).map(([k, v]) => (
                      <div key={k} className="rounded-2xl border border-slate-100 bg-slate-50 px-4 py-3 text-sm">
                        <span className="text-slate-400">{k}</span>
                        <span className="ml-2 font-semibold text-slate-800">{v}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}
          </div>
        )

      // ── Comparaison ────────────────────────────────────────────────────────
      case 'comparaison':
        return (
          <div className="space-y-6">
            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <h2 className="text-xl font-semibold text-slate-900">Comparaison des modèles</h2>
              <p className="mt-1 text-sm text-slate-500">Top 8 runs triés par R².</p>
            </div>
            {runsLoading ? <Spinner /> : runsError ? <ErrorBox message={runsError} /> : (
              <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={comparisonChartData} margin={{ top: 10, right: 10, left: 0, bottom: 40 }}>
                      <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" />
                      <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#64748b' }} angle={-30} textAnchor="end" />
                      <YAxis tick={{ fontSize: 11, fill: '#64748b' }} />
                      <Tooltip formatter={v => v.toFixed(4)} />
                      <Legend />
                      <Bar dataKey="r2"   fill="#10b981" radius={[8, 8, 0, 0]} name="R²" />
                      <Bar dataKey="rmse" fill="#2563eb" radius={[8, 8, 0, 0]} name="RMSE" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </div>
        )

      // ── Prédiction ─────────────────────────────────────────────────────────
      case 'prediction':
        return (
          <div className="space-y-6">
            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <h2 className="text-xl font-semibold text-slate-900">Prédiction</h2>
              <p className="mt-1 text-sm text-slate-500">Remplis les caractéristiques du bâtiment pour prédire la charge de chauffage.</p>
            </div>

            <div className="grid gap-6 lg:grid-cols-2">
              <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
                <h3 className="text-sm font-semibold text-slate-900 mb-4">Caractéristiques du bâtiment</h3>
                <div className="grid gap-4 sm:grid-cols-2">
                  {Object.entries(form).map(([key, val]) => (
                    <label key={key} className="space-y-1 text-sm text-slate-600">
                      {key.replace(/_/g, ' ')}
                      <input
                        type="number"
                        step="any"
                        value={val}
                        onChange={e => setForm(p => ({ ...p, [key]: parseFloat(e.target.value) || 0 }))}
                        className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-slate-900 focus:border-emerald-400 focus:outline-none"
                      />
                    </label>
                  ))}
                </div>
                <button
                  onClick={handlePredict}
                  disabled={predLoading}
                  className="mt-6 w-full rounded-2xl bg-emerald-600 px-4 py-3 text-sm font-semibold text-white hover:bg-emerald-700 disabled:opacity-50"
                >
                  {predLoading ? 'Calcul...' : 'Prédire la charge de chauffage'}
                </button>
              </div>

              <div className="rounded-3xl border border-slate-200 bg-slate-50 p-6 shadow-sm flex flex-col items-center justify-center">
                {prediction !== null ? (
                  <>
                    <p className="text-sm text-slate-500 mb-2">Charge de chauffage prédite</p>
                    <p className="text-6xl font-bold text-emerald-600">{prediction}</p>
                    <p className="mt-2 text-slate-400 text-sm">kWh/m²</p>
                  </>
                ) : (
                  <p className="text-slate-400 text-sm text-center">Lance une prédiction pour voir le résultat ici.</p>
                )}
              </div>
            </div>
          </div>
        )

      // ── Historique des runs ────────────────────────────────────────────────
      case 'historique':
        return (
          <div className="space-y-6">
            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <h2 className="text-xl font-semibold text-slate-900">Historique des runs MLflow</h2>
            </div>
            {runsLoading ? <Spinner /> : runsError ? <ErrorBox message={runsError} /> : (
              <div className="space-y-3">
                {runs.map(run => (
                  <div key={run.run_id} className="rounded-3xl border border-slate-200 bg-slate-50 p-5">
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                      <div>
                        <p className="text-sm font-semibold text-slate-900">{run.model}</p>
                        <p className="text-xs text-slate-400">ID : {run.id} · {run.status}</p>
                      </div>
                      <div className="flex gap-3 text-sm">
                        <span className="rounded-full bg-emerald-50 px-3 py-1 text-emerald-700">R² {run.r2}</span>
                        <span className="rounded-full bg-slate-100 px-3 py-1 text-slate-600">MAE {run.mae}</span>
                        <span className="rounded-full bg-slate-100 px-3 py-1 text-slate-600">RMSE {run.rmse}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )

      // ── Versions (Registry) ────────────────────────────────────────────────
      case 'versions':
        return (
          <div className="space-y-6">
            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <h2 className="text-xl font-semibold text-slate-900">Model Registry</h2>
              <p className="mt-1 text-sm text-slate-500">Versions enregistrées dans MLflow Registry.</p>
            </div>
            {regLoading ? <Spinner /> : regError ? <ErrorBox message={regError} /> : (
              <div className="space-y-3">
                {(registryData?.versions ?? []).map(v => (
                  <div key={v.version} className="rounded-3xl border border-slate-200 bg-slate-50 p-5 flex items-center justify-between">
                    <div>
                      <p className="text-sm font-semibold text-slate-900">Version {v.version}</p>
                      <p className="text-xs text-slate-400">Run : {v.run_id}</p>
                    </div>
                    <span className={`rounded-full px-3 py-1 text-xs font-semibold ${
                      v.stage === 'Production' ? 'bg-emerald-100 text-emerald-700' :
                      v.stage === 'Staging'    ? 'bg-amber-100 text-amber-700' :
                      'bg-slate-100 text-slate-600'
                    }`}>
                      {v.stage}
                    </span>
                  </div>
                ))}
                {(registryData?.versions ?? []).length === 0 && (
                  <p className="text-sm text-slate-400">Aucune version enregistrée. Lance register_model.py d'abord.</p>
                )}
              </div>
            )}
          </div>
        )

      // ── Drift Monitor ──────────────────────────────────────────────────────
      case 'drift':
        return (
          <div className="space-y-6">
            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <h2 className="text-xl font-semibold text-slate-900">Drift Monitor</h2>
              <p className="mt-1 text-sm text-slate-500">Résultats du dernier run de détection de drift.</p>
            </div>
            {driftLoading ? <Spinner /> : driftError ? <ErrorBox message={driftError} /> : driftData?.drift_share == null ? (
              <p className="text-sm text-slate-400">Aucune donnée de drift. Lance detect_drift.py d'abord.</p>
            ) : (
              <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-4">
                <StatCard label="Drift share"       value={`${(driftData.drift_share * 100).toFixed(1)}%`} />
                <StatCard label="Colonnes driftées" value={driftData.drifted_columns} sub={`sur ${driftData.total_columns} total`} />
                <StatCard label="Ré-entraînement"   value={driftData.retrain_triggered ? 'Déclenché' : 'Non requis'} />
                <StatCard label="Statut"            value={driftData.drift_share > 0.30 ? '🔴 Critique' : driftData.drift_share > 0.15 ? '🟡 Alerte' : '🟢 Stable'} />
              </div>
            )}
          </div>
        )

      default:
        return null
    }
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <div className="grid min-h-screen grid-cols-[240px_1fr] grid-rows-[72px_1fr]">

        {/* Header */}
        <header className="col-span-2 flex items-center justify-between gap-4 border-b border-slate-200 bg-white px-6 shadow-sm">
          <div className="flex items-center gap-3 text-sm font-semibold text-slate-900">
            <span className="inline-flex h-10 w-10 items-center justify-center rounded-3xl bg-emerald-600 text-xl text-white">ML</span>
            <div>
              <p>Projet ML</p>
              <p className="text-xs text-slate-500">Dashboard MLOps — données en temps réel</p>
            </div>
          </div>
          <div className="flex items-center gap-3 text-sm">
            <span className="rounded-full bg-emerald-50 px-3 py-1 text-emerald-700 text-xs">API connectée</span>
            <span className="rounded-full bg-slate-100 px-3 py-1 text-slate-600">v2.0</span>
          </div>
        </header>

        {/* Sidebar */}
        <aside className="flex h-full flex-col gap-6 border-r border-slate-200 bg-white p-4">
          {['Navigation', 'MLOps'].map(group => (
            <div key={group}>
              <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">{group}</p>
              <div className="mt-3 space-y-1">
                {sections.filter(s => s.group === group).map(item => (
                  <button
                    key={item.id}
                    onClick={() => setActiveSection(item.id)}
                    className={`flex w-full items-center gap-3 rounded-2xl px-4 py-3 text-left text-sm transition ${
                      activeSection === item.id ? 'bg-emerald-50 text-emerald-700' : 'text-slate-600 hover:bg-slate-100'
                    }`}
                  >
                    <span className="inline-flex h-3.5 w-3.5 rounded-full bg-slate-300" />
                    {item.label}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </aside>

        {/* Main */}
        <main className="space-y-6 overflow-y-auto bg-slate-50 p-6">
          {renderSection()}
        </main>
      </div>

      {/* Notifications */}
      <div className="fixed bottom-5 right-5 z-20 flex flex-col gap-3">
        {notifications.map(note => (
          <div key={note.id} className={`max-w-sm rounded-3xl border px-4 py-3 text-sm shadow-lg transition ${
            note.type === 'success' ? 'border-emerald-200 bg-emerald-50 text-emerald-700' : 'border-slate-200 bg-white text-slate-800'
          }`}>
            {note.text}
          </div>
        ))}
      </div>
    </div>
  )
}
