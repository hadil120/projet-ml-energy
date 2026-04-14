import { useEffect, useMemo, useState } from 'react'
import { AreaChart, Area, BarChart, Bar, LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts'

const modelDefinitions = [
  { id: 'svm', label: 'SVM', description: 'Frontière de décision pour petites données.', color: 'bg-sky-500', badge: 'Classique' },
  { id: 'rf', label: 'Random Forest', description: 'Ensemble d’arbres pour robustesse.', color: 'bg-emerald-500', badge: 'Fiable' },
  { id: 'knn', label: 'KNN', description: 'Instance-based simple et rapide.', color: 'bg-amber-500', badge: 'Interprétable' },
  { id: 'lr', label: 'Logistic Regression', description: 'Régression logistique pour classification.', color: 'bg-slate-600', badge: 'Léger' },
  { id: 'nn', label: 'Neural Network', description: 'Réseau de neurones pour modèles complexes.', color: 'bg-orange-500', badge: 'Avancé' }
]

const sampleRows = [
  { chauffage: 15.8, clim: 10.3, surface: 750, orientation: 'S', ep: 0.72 },
  { chauffage: 16.1, clim: 11.2, surface: 650, orientation: 'N', ep: 0.81 },
  { chauffage: 19.8, clim: 13.5, surface: 820, orientation: 'E', ep: 0.93 },
  { chauffage: 14.3, clim: 9.9, surface: 710, orientation: 'S', ep: 0.68 },
  { chauffage: 18.2, clim: 12.1, surface: 830, orientation: 'W', ep: 0.89 },
  { chauffage: 17.0, clim: 11.7, surface: 780, orientation: 'N', ep: 0.75 }
]

const versions = [
  { id: 'v2.4', model: 'Random Forest', dataset: 'energy-v2', accuracy: '92.8%', date: '12 avr. 2026' },
  { id: 'v2.3', model: 'Neural Network', dataset: 'energy-v2', accuracy: '91.4%', date: '08 avr. 2026' },
  { id: 'v2.2', model: 'SVM', dataset: 'energy-v1', accuracy: '89.0%', date: '02 avr. 2026' }
]

const dashboardTrend = [
  { name: 'Lun', score: 78 },
  { name: 'Mar', score: 84 },
  { name: 'Mer', score: 88 },
  { name: 'Jeu', score: 91 },
  { name: 'Ven', score: 90 },
  { name: 'Sam', score: 93 }
]

const rocData = [
  { name: '0.0', roc: 0.0, pr: 0.0 },
  { name: '0.2', roc: 0.3, pr: 0.2 },
  { name: '0.4', roc: 0.6, pr: 0.5 },
  { name: '0.6', roc: 0.8, pr: 0.75 },
  { name: '0.8', roc: 0.95, pr: 0.88 },
  { name: '1.0', roc: 1.0, pr: 1.0 }
]

const confusionData = [
  { label: 'TP', value: 80, type: 'True Positive' },
  { label: 'FP', value: 12, type: 'False Positive' },
  { label: 'FN', value: 9, type: 'False Negative' },
  { label: 'TN', value: 71, type: 'True Negative' }
]

const experiments = [
  { id: 'exp-102', model: 'Random Forest', score: '0.92', status: 'Terminé', date: '13 avr. 2026' },
  { id: 'exp-101', model: 'Neural Network', score: '0.91', status: 'Terminé', date: '11 avr. 2026' },
  { id: 'exp-100', model: 'SVM', score: '0.89', status: 'Échoué', date: '09 avr. 2026' }
]

const sections = [
  { id: 'dashboard', label: 'Dashboard', group: 'Navigation' },
  { id: 'entrainement', label: 'Entraînement', group: 'Navigation' },
  { id: 'resultats', label: 'Résultats', group: 'Navigation' },
  { id: 'donnees', label: 'Données', group: 'Navigation' },
  { id: 'comparaison', label: 'Comparaison', group: 'Navigation' },
  { id: 'historique', label: 'Historique', group: 'MLOps' },
  { id: 'versions', label: 'Versions', group: 'MLOps' },
  { id: 'parametres', label: 'Paramètres', group: 'MLOps' }
]

export default function App() {
  const [activeSection, setActiveSection] = useState('dashboard')
  const [selectedModels, setSelectedModels] = useState(['rf'])
  const [activeModel, setActiveModel] = useState('rf')
  const [trainingSource, setTrainingSource] = useState('scratch')
  const [autoTuning, setAutoTuning] = useState(false)
  const [hyperparams, setHyperparams] = useState({ learningRate: 0.02, maxDepth: 10, nEstimators: 120, kNeighbors: 5 })
  const [savedConfigs, setSavedConfigs] = useState([])
  const [visibleColumns, setVisibleColumns] = useState(['chauffage', 'clim', 'surface', 'orientation', 'ep'])
  const [uploadedFile, setUploadedFile] = useState(null)
  const [notifications, setNotifications] = useState([{ id: 1, text: 'Interface chargée', type: 'success' }])

  const activeModelData = useMemo(() => modelDefinitions.find((item) => item.id === activeModel) ?? modelDefinitions[0], [activeModel])

  useEffect(() => {
    if (!notifications.length) return
    const timer = setTimeout(() => setNotifications((prev) => prev.slice(1)), 3000)
    return () => clearTimeout(timer)
  }, [notifications])

  const notify = (text, type = 'info') => {
    setNotifications((prev) => [...prev, { id: Date.now(), text, type }])
  }

  const toggleModel = (id) => {
    setSelectedModels((prev) => {
      const next = prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
      return next.length ? next : prev
    })
    setActiveModel(id)
  }

  const updateHyperparam = (key, value) => setHyperparams((prev) => ({ ...prev, [key]: value }))

  const saveConfig = () => {
    setSavedConfigs((prev) => [
      { id: `cfg-${Date.now()}`, name: `${activeModelData.label} - ${new Date().toLocaleTimeString()}`, model: activeModelData.label, params: { ...hyperparams, trainingSource, autoTuning } },
      ...prev
    ])
    notify('Configuration enregistrée.', 'success')
  }

  const runTraining = () => {
    notify(`Entraînement de ${activeModelData.label} lancé.`, 'success')
    setActiveSection('resultats')
  }

  const toggleColumn = (column) => {
    setVisibleColumns((prev) => (prev.includes(column) ? prev.filter((item) => item !== column) : [...prev, column]))
  }

  const handleUpload = (event) => {
    const file = event.target.files?.[0]
    if (!file) return
    setUploadedFile(file.name)
    notify(`Fichier ${file.name} chargé.`, 'success')
  }

  const cleanData = () => notify('Valeurs manquantes supprimées.', 'success')
  const toggleAutoTuning = () => {
    setAutoTuning((prev) => !prev)
    notify(`Tuning automatique ${!autoTuning ? 'activé' : 'désactivé'}.`, 'info')
  }

  const exportReport = (type) => notify(`${type} exporté.`, 'success')
  const rollbackVersion = (version) => notify(`Rollback vers ${version} effectué.`, 'success')

  const modelCount = selectedModels.length
  const performanceSummary = useMemo(
    () => selectedModels.map((id) => {
      const model = modelDefinitions.find((item) => item.id === id) ?? modelDefinitions[0]
      return {
        ...model,
        accuracy: `${(0.88 + Math.random() * 0.08).toFixed(3)}`,
        f1: `${(0.80 + Math.random() * 0.12).toFixed(3)}`
      }
    }),
    [selectedModels]
  )

  const comparisonChartData = useMemo(
    () => performanceSummary.map((model) => ({ name: model.label, accuracy: Number(model.accuracy), f1: Number(model.f1) })),
    [performanceSummary]
  )

  const renderSection = () => {
    switch (activeSection) {
      case 'entrainement':
        return (
          <div className="space-y-6">
            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                <div>
                  <h2 className="text-xl font-semibold text-slate-900">Entraînement</h2>
                  <p className="mt-1 text-sm text-slate-500">Choisis ton algorithme, configure les hyperparamètres et lance la session.</p>
                </div>
                <div className="flex flex-col gap-3 sm:flex-row">
                  <button onClick={toggleAutoTuning} className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-2 text-sm text-slate-700 hover:bg-slate-100">Tuning automatique</button>
                  <button onClick={runTraining} className="rounded-2xl bg-emerald-600 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-700">Lancer l’entraînement</button>
                </div>
              </div>
            </div>

            <div className="grid gap-6 xl:grid-cols-[0.9fr_1.1fr]">
              <div className="space-y-6 rounded-3xl border border-slate-200 bg-slate-50 p-6">
                <div className="flex items-center justify-between gap-4">
                  <h3 className="text-base font-semibold text-slate-900">Sélection des modèles</h3>
                  <span className="rounded-full bg-slate-100 px-3 py-1 text-xs uppercase tracking-[0.18em] text-slate-500">{modelCount} sélectionné{modelCount > 1 ? 's' : ''}</span>
                </div>
                <div className="space-y-3">
                  {modelDefinitions.map((model) => (
                    <button
                      key={model.id}
                      onClick={() => toggleModel(model.id)}
                      className={`flex w-full items-center justify-between gap-4 rounded-3xl border p-4 text-left transition ${selectedModels.includes(model.id) ? 'border-emerald-300 bg-emerald-50' : 'border-slate-200 bg-white hover:border-slate-300'}`}
                    >
                      <div>
                        <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                          <span className={`inline-flex h-3.5 w-3.5 rounded-full ${model.color}`} />
                          {model.label}
                        </div>
                        <p className="mt-1 text-xs text-slate-500">{model.description}</p>
                      </div>
                      <span className="rounded-full bg-slate-100 px-3 py-1 text-[11px] uppercase tracking-[0.18em] text-slate-600">{model.badge}</span>
                    </button>
                  ))}
                </div>
              </div>

              <div className="rounded-3xl border border-slate-200 bg-white p-6">
                <div className="flex items-center justify-between gap-4">
                  <div>
                    <h3 className="text-base font-semibold text-slate-900">Paramétrage avancé</h3>
                    <p className="text-sm text-slate-500">Applique les hyperparamètres et sauvegarde tes configurations.</p>
                  </div>
                  <span className="rounded-full bg-slate-100 px-3 py-1 text-xs uppercase tracking-[0.18em] text-slate-500">{trainingSource === 'scratch' ? 'From scratch' : 'Pré-entraîné'}</span>
                </div>
                <div className="mt-6 space-y-4">
                  <div className="grid gap-4 sm:grid-cols-2">
                    <label className="space-y-2 text-sm text-slate-600">
                      Learning rate
                      <input type="range" min="0.001" max="0.2" step="0.001" value={hyperparams.learningRate} onChange={(e) => updateHyperparam('learningRate', Number(e.target.value))} className="w-full" />
                      <span className="text-xs text-slate-500">{hyperparams.learningRate.toFixed(3)}</span>
                    </label>
                    <label className="space-y-2 text-sm text-slate-600">
                      Max depth
                      <input type="range" min="1" max="30" value={hyperparams.maxDepth} onChange={(e) => updateHyperparam('maxDepth', Number(e.target.value))} className="w-full" />
                      <span className="text-xs text-slate-500">{hyperparams.maxDepth}</span>
                    </label>
                  </div>

                  <div className="grid gap-4 sm:grid-cols-2">
                    <label className="space-y-2 text-sm text-slate-600">
                      Nombre d’arbres
                      <input type="range" min="10" max="300" value={hyperparams.nEstimators} onChange={(e) => updateHyperparam('nEstimators', Number(e.target.value))} className="w-full" />
                      <span className="text-xs text-slate-500">{hyperparams.nEstimators}</span>
                    </label>
                    <label className="space-y-2 text-sm text-slate-600">
                      K pour KNN
                      <input type="range" min="1" max="25" value={hyperparams.kNeighbors} onChange={(e) => updateHyperparam('kNeighbors', Number(e.target.value))} className="w-full" />
                      <span className="text-xs text-slate-500">{hyperparams.kNeighbors}</span>
                    </label>
                  </div>

                  <div className="grid gap-3 sm:grid-cols-2">
                    <button onClick={saveConfig} className="rounded-2xl bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800">Sauvegarder la configuration</button>
                    <button onClick={() => setTrainingSource(trainingSource === 'scratch' ? 'pretrained' : 'scratch')} className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-2 text-sm text-slate-700 hover:bg-slate-100">
                      {trainingSource === 'scratch' ? 'Charger modèle pré-entraîné' : 'Entraîner depuis zéro'}
                    </button>
                  </div>
                </div>
                {savedConfigs.length > 0 && (
                  <div className="mt-6 rounded-3xl border border-slate-100 bg-slate-50 p-4">
                    <div className="mb-3 flex items-center justify-between">
                      <h4 className="text-sm font-semibold text-slate-900">Configurations sauvegardées</h4>
                      <span className="text-xs text-slate-500">{savedConfigs.length}</span>
                    </div>
                    <div className="space-y-2">
                      {savedConfigs.slice(0, 4).map((config) => (
                        <div key={config.id} className="rounded-2xl border border-slate-200 bg-white p-3 text-sm text-slate-700">{config.name}</div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )

      case 'resultats':
        return (
          <div className="space-y-6">
            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                <div>
                  <h2 className="text-xl font-semibold text-slate-900">Résultats</h2>
                  <p className="mt-1 text-sm text-slate-500">Visualisations dynamiques et export des performances.</p>
                </div>
                <div className="flex flex-wrap gap-3">
                  <button onClick={() => exportReport('PNG')} className="rounded-2xl bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800">Exporter PNG</button>
                  <button onClick={() => exportReport('CSV')} className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-2 text-sm text-slate-700 hover:bg-slate-100">Exporter CSV</button>
                </div>
              </div>
            </div>

            <div className="grid gap-6 xl:grid-cols-[0.95fr_1.05fr]">
              <div className="space-y-6 rounded-3xl border border-slate-200 bg-slate-50 p-6">
                <div className="grid gap-4 sm:grid-cols-3">
                  <div className="rounded-3xl bg-white p-5 shadow-sm">
                    <p className="text-sm text-slate-500">Accuracy</p>
                    <p className="mt-3 text-3xl font-semibold text-slate-900">91.8%</p>
                  </div>
                  <div className="rounded-3xl bg-white p-5 shadow-sm">
                    <p className="text-sm text-slate-500">F1-score</p>
                    <p className="mt-3 text-3xl font-semibold text-slate-900">0.89</p>
                  </div>
                  <div className="rounded-3xl bg-white p-5 shadow-sm">
                    <p className="text-sm text-slate-500">AUC ROC</p>
                    <p className="mt-3 text-3xl font-semibold text-slate-900">0.94</p>
                  </div>
                </div>

                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="rounded-3xl bg-white p-5 shadow-sm">
                    <div className="mb-3 flex items-center justify-between">
                      <div>
                        <h3 className="text-sm font-semibold text-slate-900">Matrice de confusion</h3>
                        <p className="text-xs text-slate-500">Comptes TP / FP / FN / TN</p>
                      </div>
                      <span className="text-xs text-slate-500">interactive</span>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      {confusionData.map((cell) => (
                        <div key={cell.label} className="rounded-3xl border border-slate-200 bg-slate-100 p-4 transition hover:border-emerald-300 hover:bg-emerald-50">
                          <p className="text-xs uppercase tracking-[0.18em] text-slate-500">{cell.type}</p>
                          <p className="mt-3 text-2xl font-semibold text-slate-900">{cell.value}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="rounded-3xl bg-white p-5 shadow-sm">
                    <div className="mb-3 flex items-center justify-between">
                      <div>
                        <h3 className="text-sm font-semibold text-slate-900">Courbes ROC / PR</h3>
                        <p className="text-xs text-slate-500">Sensibilité du modèle selon le seuil</p>
                      </div>
                      <span className="text-xs text-slate-500">interactive</span>
                    </div>
                    <div className="h-56">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={rocData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                          <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" />
                          <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#64748b' }} />
                          <YAxis tick={{ fontSize: 11, fill: '#64748b' }} />
                          <Tooltip formatter={(value) => value.toFixed(2)} />
                          <Legend verticalAlign="top" height={24} />
                          <Line type="monotone" dataKey="roc" stroke="#10b981" strokeWidth={3} dot={{ r: 3 }} />
                          <Line type="monotone" dataKey="pr" stroke="#2563eb" strokeWidth={3} dot={{ r: 3 }} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              </div>

              <div className="rounded-3xl border border-slate-200 bg-white p-6">
                <div className="flex items-center justify-between gap-4">
                  <div>
                    <h3 className="text-base font-semibold text-slate-900">Export et rapports</h3>
                    <p className="text-sm text-slate-500">Sauvegarde les résultats et partage-les facilement.</p>
                  </div>
                </div>
                <div className="mt-6 space-y-4">
                  <button onClick={() => exportReport('CSV')} className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-left text-sm text-slate-700 hover:bg-slate-100">Exporter les métriques CSV</button>
                  <div className="rounded-3xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-500">Le dashboard présente des graphiques interactifs et des exports prêts à partager.</div>
                </div>
              </div>
            </div>
          </div>
        )

      case 'donnees':
        return (
          <div className="space-y-6">
            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <h2 className="text-xl font-semibold text-slate-900">Données</h2>
              <p className="mt-1 text-sm text-slate-500">Prévisualise le dataset, filtre les colonnes et nettoie rapidement.</p>
            </div>

            <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
              <div className="rounded-3xl border border-slate-200 bg-slate-50 p-6">
                <div className="rounded-3xl border border-dashed border-slate-300 bg-white p-8 text-center text-slate-500">
                  <p className="text-4xl">📤</p>
                  <p className="mt-4 text-base font-semibold text-slate-900">Importer un dataset</p>
                  <p className="mt-2 text-sm text-slate-500">Format CSV ou Excel · max 10 Mo</p>
                  <label className="mt-6 inline-flex cursor-pointer items-center rounded-2xl bg-emerald-600 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-700">
                    Sélectionner un fichier
                    <input type="file" accept=".csv,.xlsx" onChange={handleUpload} className="sr-only" />
                  </label>
                  {uploadedFile && <p className="mt-3 text-sm text-slate-500">Fichier chargé: {uploadedFile}</p>}
                </div>

                <div className="mt-6 rounded-3xl border border-slate-200 bg-white p-5">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-slate-900">Colonnes visibles</h3>
                    <button onClick={cleanData} className="rounded-2xl bg-slate-100 px-3 py-1 text-xs text-slate-700 hover:bg-slate-200">Nettoyage rapide</button>
                  </div>
                  <div className="mt-4 flex flex-wrap gap-2">
                    {['chauffage', 'clim', 'surface', 'orientation', 'ep'].map((col) => (
                      <button
                        key={col}
                        onClick={() => toggleColumn(col)}
                        className={`rounded-full px-3 py-1 text-xs font-medium transition ${visibleColumns.includes(col) ? 'bg-emerald-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}
                      >
                        {col}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm overflow-x-auto">
                <div className="flex items-center justify-between gap-2">
                  <h3 className="text-base font-semibold text-slate-900">Aperçu du dataset</h3>
                  <span className="text-xs uppercase tracking-[0.18em] text-slate-500">6 lignes</span>
                </div>
                <table className="mt-5 min-w-full text-sm">
                  <thead className="bg-slate-50 text-slate-500">
                    <tr>
                      {visibleColumns.map((col) => (
                        <th key={col} className="border-b border-slate-200 px-4 py-3 text-left uppercase tracking-[0.08em]">{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="bg-white">
                    {sampleRows.map((row, index) => (
                      <tr key={index} className={index % 2 === 0 ? 'bg-slate-50' : ''}>
                        {visibleColumns.map((col) => (
                          <td key={col} className="border-b border-slate-200 px-4 py-3 text-slate-700">{row[col] ?? '-'}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )

      case 'comparaison':
        return (
          <div className="space-y-6">
            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                <div>
                  <h2 className="text-xl font-semibold text-slate-900">Comparaison</h2>
                  <p className="mt-1 text-sm text-slate-500">Analyse côte à côte des performances des modèles sélectionnés.</p>
                </div>
                <span className="rounded-full bg-slate-100 px-3 py-1 text-xs uppercase tracking-[0.18em] text-slate-500">{selectedModels.length} modèle{selectedModels.length > 1 ? 's' : ''}</span>
              </div>
            </div>

            <div className="space-y-6">
              <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={comparisonChartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                      <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" />
                      <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#64748b' }} />
                      <YAxis tick={{ fontSize: 11, fill: '#64748b' }} />
                      <Tooltip formatter={(value) => value.toFixed(2)} />
                      <Legend />
                      <Bar dataKey="accuracy" fill="#10b981" radius={[8, 8, 0, 0]} />
                      <Bar dataKey="f1" fill="#2563eb" radius={[8, 8, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="grid gap-4 lg:grid-cols-2">
                {performanceSummary.map((model) => (
                  <div key={model.id} className="rounded-3xl border border-slate-200 bg-slate-50 p-5 shadow-sm">
                    <div className="flex items-center justify-between gap-4">
                      <div>
                        <p className="text-sm font-semibold text-slate-900">{model.label}</p>
                        <p className="mt-1 text-xs text-slate-500">{model.description}</p>
                      </div>
                      <span className={`inline-flex h-3.5 w-3.5 rounded-full ${model.color}`} />
                    </div>
                    <div className="mt-5 grid gap-3 rounded-3xl bg-white p-4 shadow-sm">
                      <div className="flex items-center justify-between text-sm text-slate-500">
                        <span>Accuracy</span>
                        <span className="font-semibold text-slate-900">{model.accuracy}</span>
                      </div>
                      <div className="flex items-center justify-between text-sm text-slate-500">
                        <span>F1-score</span>
                        <span className="font-semibold text-slate-900">{model.f1}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )

      case 'historique':
        return (
          <div className="space-y-6">
            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <h2 className="text-xl font-semibold text-slate-900">Historique des expérimentations</h2>
              <p className="mt-1 text-sm text-slate-500">Suivi des runs, des scores et des états.</p>
            </div>
            <div className="space-y-4">
              {experiments.map((experiment) => (
                <div key={experiment.id} className="rounded-3xl border border-slate-200 bg-slate-50 p-5 shadow-sm">
                  <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                    <div>
                      <p className="text-sm font-semibold text-slate-900">{experiment.model} · {experiment.id}</p>
                      <p className="text-xs text-slate-500">{experiment.date} · {experiment.status}</p>
                    </div>
                    <div className="flex items-center gap-2 text-sm">
                      <span className="rounded-full bg-slate-100 px-3 py-1 text-slate-600">Score {experiment.score}</span>
                      <button onClick={() => setActiveSection('resultats')} className="rounded-2xl bg-emerald-600 px-3 py-1 text-xs font-semibold text-white hover:bg-emerald-700">Voir</button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )

      case 'versions':
        return (
          <div className="space-y-6">
            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <h2 className="text-xl font-semibold text-slate-900">Versions</h2>
              <p className="mt-1 text-sm text-slate-500">Versions des modèles et datasets suivis.</p>
            </div>
            <div className="grid gap-4">
              {versions.map((version) => (
                <div key={version.id} className="flex flex-col gap-3 rounded-3xl border border-slate-200 bg-slate-50 p-5 sm:flex-row sm:items-center sm:justify-between">
                  <div>
                    <p className="text-sm font-semibold text-slate-900">{version.id} · {version.model}</p>
                    <p className="text-xs text-slate-500">Dataset {version.dataset} · {version.date}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="rounded-full bg-white px-3 py-1 text-xs text-slate-600">{version.accuracy}</span>
                    <button onClick={() => rollbackVersion(version.id)} className="rounded-2xl border border-slate-200 bg-white px-3 py-1 text-xs text-slate-700 hover:bg-slate-100">Rollback</button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )

      case 'parametres':
        return (
          <div className="space-y-6">
            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <h2 className="text-xl font-semibold text-slate-900">Paramètres</h2>
              <p className="mt-1 text-sm text-slate-500">Options backend, export et tutoriels intégrés.</p>
            </div>
            <div className="grid gap-4 lg:grid-cols-2">
              <div className="rounded-3xl border border-slate-200 bg-slate-50 p-6">
                <h3 className="text-sm font-semibold text-slate-900">Export modèle</h3>
                <p className="mt-2 text-sm text-slate-500">Exporter le modèle entraîné en pickle/joblib pour le backend Python.</p>
                <div className="mt-4 flex flex-col gap-3">
                  <button onClick={() => exportReport('pickle')} className="rounded-2xl bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800">Exporter Pickle</button>
                  <button onClick={() => exportReport('joblib')} className="rounded-2xl border border-slate-200 bg-white px-4 py-2 text-sm text-slate-700 hover:bg-slate-100">Exporter Joblib</button>
                </div>
              </div>
              <div className="rounded-3xl border border-slate-200 bg-slate-50 p-6">
                <h3 className="text-sm font-semibold text-slate-900">Guide utilisateur</h3>
                <p className="mt-2 text-sm text-slate-500">Clique sur les icônes d’aide pour les tooltips et suis les étapes de training.</p>
                <ul className="mt-4 space-y-3 text-sm text-slate-600">
                  <li>1. Sélectionne un modèle et ses hyperparamètres.</li>
                  <li>2. Charge ou nettoie tes données.</li>
                  <li>3. Lance l’entraînement et exporte les résultats.</li>
                </ul>
              </div>
            </div>
          </div>
        )

      default:
        return (
          <div className="space-y-6">
            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                <div>
                  <h1 className="text-2xl font-semibold text-slate-900">Tableau de bord</h1>
                  <p className="mt-1 text-sm text-slate-500">Vue globale de l’entraînement, des performances et des versions.</p>
                </div>
                <div className="flex flex-wrap items-center gap-3 text-sm text-slate-600">
                  <span className="rounded-full bg-slate-100 px-3 py-1">Modèle actif : {activeModelData.label}</span>
                  <span className="rounded-full bg-slate-100 px-3 py-1">Mode : {trainingSource === 'scratch' ? 'From scratch' : 'Pré-entraîné'}</span>
                </div>
              </div>
            </div>

            <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-4">
              <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
                <p className="text-sm text-slate-500">Modèles sélectionnés</p>
                <p className="mt-3 text-3xl font-semibold text-slate-900">{selectedModels.length}</p>
              </div>
              <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
                <p className="text-sm text-slate-500">Configurations sauvegardées</p>
                <p className="mt-3 text-3xl font-semibold text-slate-900">{savedConfigs.length}</p>
              </div>
              <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
                <p className="text-sm text-slate-500">Experiments</p>
                <p className="mt-3 text-3xl font-semibold text-slate-900">{experiments.length}</p>
              </div>
              <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
                <p className="text-sm text-slate-500">Versions suivies</p>
                <p className="mt-3 text-3xl font-semibold text-slate-900">{versions.length}</p>
              </div>
            </div>

            <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
              <div className="rounded-3xl border border-slate-200 bg-slate-50 p-6 shadow-sm">
                <h2 className="text-base font-semibold text-slate-900">Performance historique</h2>
                <div className="mt-5 h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={dashboardTrend} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                      <defs>
                        <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#10b981" stopOpacity={0.35} />
                          <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" />
                      <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#64748b' }} />
                      <YAxis tick={{ fontSize: 11, fill: '#64748b' }} />
                      <Tooltip formatter={(value) => `${value}%`} />
                      <Area type="monotone" dataKey="score" stroke="#10b981" fill="url(#colorScore)" strokeWidth={3} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
                <h2 className="text-base font-semibold text-slate-900">Actions rapides</h2>
                <div className="mt-5 space-y-3">
                  <button onClick={() => setActiveSection('donnees')} className="w-full rounded-2xl bg-slate-900 px-4 py-3 text-sm font-semibold text-white hover:bg-slate-800">Gérer les données</button>
                  <button onClick={() => setActiveSection('entrainement')} className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700 hover:bg-slate-100">Paramétrer l’entraînement</button>
                </div>
                <div className="mt-6 rounded-3xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-600">Ouvre un modèle, compare les scores et exporte les résultats en un clic.</div>
              </div>
            </div>
          </div>
        )
    }
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <div className="grid min-h-screen grid-cols-[240px_1fr] grid-rows-[72px_1fr]">
        <header className="col-span-2 flex items-center justify-between gap-4 border-b border-slate-200 bg-white px-6 shadow-sm">
          <div className="flex items-center gap-3 text-sm font-semibold text-slate-900">
            <span className="inline-flex h-10 w-10 items-center justify-center rounded-3xl bg-emerald-600 text-xl text-white">ML</span>
            <div>
              <p>Projet ML</p>
              <p className="text-xs text-slate-500">Tableau de bord & gestion des modèles</p>
            </div>
          </div>
          <div className="flex items-center gap-4 text-sm text-slate-600">
            <span className="rounded-full bg-slate-100 px-3 py-1">v2.0</span>
            <button className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-slate-700 hover:bg-slate-100">Notifications</button>
          </div>
        </header>

        <aside className="flex h-full flex-col gap-3 border-r border-slate-200 bg-white p-4" role="navigation" aria-label="Menu principal">
          <div>
            <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">Navigation</p>
            <div className="mt-3 space-y-2">
              {sections.filter((section) => section.group === 'Navigation').map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setActiveSection(item.id)}
                  className={`flex w-full items-center gap-3 rounded-2xl px-4 py-3 text-left text-sm transition ${activeSection === item.id ? 'bg-emerald-50 text-emerald-700' : 'text-slate-600 hover:bg-slate-100'}`}
                >
                  <span className="inline-flex h-3.5 w-3.5 rounded-full bg-slate-300" aria-hidden="true" />
                  {item.label}
                </button>
              ))}
            </div>
          </div>
          <div>
            <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">MLOps</p>
            <div className="mt-3 space-y-2">
              {sections.filter((section) => section.group === 'MLOps').map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setActiveSection(item.id)}
                  className={`flex w-full items-center gap-3 rounded-2xl px-4 py-3 text-left text-sm transition ${activeSection === item.id ? 'bg-emerald-50 text-emerald-700' : 'text-slate-600 hover:bg-slate-100'}`}
                >
                  <span className="inline-flex h-3.5 w-3.5 rounded-full bg-slate-300" aria-hidden="true" />
                  {item.label}
                </button>
              ))}
            </div>
          </div>
        </aside>

        <main className="space-y-6 overflow-y-auto bg-slate-50 p-6" role="main">
          {renderSection()}
        </main>
      </div>

      <div className="fixed bottom-5 right-5 z-20 flex flex-col gap-3" aria-live="polite">
        {notifications.map((note) => (
          <div key={note.id} className={`max-w-sm rounded-3xl border px-4 py-3 text-sm shadow-lg transition ${note.type === 'success' ? 'border-emerald-200 bg-emerald-50 text-emerald-700' : 'border-slate-200 bg-white text-slate-800'}`}>
            {note.text}
          </div>
        ))}
      </div>
    </div>
  )
}
