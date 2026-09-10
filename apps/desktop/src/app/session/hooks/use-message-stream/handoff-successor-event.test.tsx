import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeGatewayProfile } from '@/store/profile'
import { $activeSessionStoredIdRotation, setActiveSessionStoredIdRotation } from '@/store/session'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

let handleEvent: ((event: RpcEvent) => void) | null = null
let queryClient: QueryClient

function Harness() {
  const activeSessionIdRef = useRef<string | null>('runtime-source')
  const sessionStateByRuntimeIdRef = useRef(new Map<string, ClientSessionState>())
  const stream = useMessageStream({
    activeGatewayProfile: 'compass',
    activeSessionIdRef,
    hydrateFromStoredSession: vi.fn(async () => undefined),
    queryClient,
    refreshHermesConfig: vi.fn(async () => undefined),
    refreshSessions: vi.fn(async () => undefined),
    sessionStateByRuntimeIdRef,
    updateSessionState: (sessionId, updater) => {
      const next = updater(sessionStateByRuntimeIdRef.current.get(sessionId) ?? createClientSessionState())
      sessionStateByRuntimeIdRef.current.set(sessionId, next)
      return next
    }
  })
  useEffect(() => {
    handleEvent = stream.handleGatewayEvent
  }, [stream.handleGatewayEvent])
  return null
}

const payload = {
  acknowledged: true,
  parent_runtime_id: 'runtime-source',
  source_session_id: 'stored-source',
  successor_session_id: 'stored-successor'
}

function send(overrides: Partial<RpcEvent> = {}) {
  act(() =>
    handleEvent!({
      type: 'handoff.successor.created',
      session_id: 'runtime-source',
      profile: 'compass',
      payload,
      ...overrides
    } as RpcEvent)
  )
}

beforeEach(async () => {
  handleEvent = null
  queryClient = new QueryClient()
  $activeGatewayProfile.set('compass')
  setActiveSessionStoredIdRotation(null)
  render(<Harness />)
  await waitFor(() => expect(handleEvent).not.toBeNull())
})

afterEach(() => {
  cleanup()
  queryClient.clear()
  setActiveSessionStoredIdRotation(null)
  $activeGatewayProfile.set('default')
  vi.restoreAllMocks()
})

describe('acknowledged handoff event', () => {
  it('publishes exact source and durable successor for the active profile', () => {
    send()
    expect($activeSessionStoredIdRotation.get()).toEqual({
      kind: 'handoff',
      profile: 'compass',
      runtimeSessionId: 'runtime-source',
      previousStoredSessionId: 'stored-source',
      nextStoredSessionId: 'stored-successor'
    })
  })

  it.each([
    { profile: 'other' },
    { profile: undefined, payload: { ...payload, profile: 'other' } },
    { session_id: 'background-runtime' },
    { payload: { ...payload, acknowledged: false } },
    { payload: { ...payload, parent_runtime_id: 'other' } },
    { payload: { ...payload, successor_session_id: '' } },
    { payload: { ...payload, successor_session_id: 'stored-source' } }
  ])('does not follow mismatched or unacknowledged events: %j', override => {
    send(override as Partial<RpcEvent>)
    expect($activeSessionStoredIdRotation.get()).toBeNull()
  })
})
