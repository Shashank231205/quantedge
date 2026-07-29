import React from 'react'
import ReactDOM from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'

import Shell from './layout/Shell'
import Dashboard from './screens/Dashboard'
import Factors from './screens/Factors'
import Backtest from './screens/Backtest'
import Risk from './screens/Risk'
import Pipeline from './screens/Pipeline'
import Analyst from './screens/Analyst'
import Inu from './screens/Inu'
import './styles.css'

const router = createBrowserRouter([
  {
    path: '/',
    element: <Shell />,
    children: [
      { index: true, element: <Dashboard /> },
      { path: 'factors', element: <Factors /> },
      { path: 'backtest', element: <Backtest /> },
      { path: 'risk', element: <Risk /> },
      { path: 'pipeline', element: <Pipeline /> },
      { path: 'analyst', element: <Analyst /> },
      { path: 'inu', element: <Inu /> },
    ],
  },
])

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
)
