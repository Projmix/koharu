'use client'

import { useEffect, useRef, useState, type ComponentProps } from 'react'

import { registerPendingEdit } from '@/lib/history'
import { Textarea } from '@koharu/ui/components/textarea'

type CommitTextareaProps = Omit<ComponentProps<typeof Textarea>, 'value' | 'onChange'> & {
  value: string
  delay?: number
  onCommit: (value: string) => void | Promise<void>
}

export function CommitTextarea({ value, delay = 360, onCommit, ...props }: CommitTextareaProps) {
  const [draft, setDraft] = useState(value)
  const timer = useRef<number | null>(null)
  const composing = useRef(false)
  const external = useRef(value)
  const draftRef = useRef(value)
  const onCommitRef = useRef(onCommit)
  const inFlight = useRef<Promise<void> | null>(null)
  const inFlightValue = useRef<string | null>(null)
  const flushRef = useRef<() => Promise<void>>(() => Promise.resolve())

  draftRef.current = draft
  onCommitRef.current = onCommit

  useEffect(() => {
    external.current = value
    if (!composing.current && timer.current === null && inFlight.current === null) {
      draftRef.current = value
      setDraft(value)
    }
  }, [value])

  useEffect(() => registerPendingEdit(() => flushRef.current()), [])

  useEffect(
    () => () => {
      if (timer.current !== null) window.clearTimeout(timer.current)
    },
    [],
  )

  const commit = (next: string): Promise<void> => {
    if (timer.current !== null) window.clearTimeout(timer.current)
    timer.current = null
    if (next === external.current) return Promise.resolve()
    if (inFlight.current && inFlightValue.current === next) return inFlight.current

    const pending = Promise.resolve()
      .then(() => onCommitRef.current(next))
      .then(() => {
        external.current = next
      })
    inFlight.current = pending
    inFlightValue.current = next
    pending.then(
      () => {
        if (inFlight.current === pending) {
          inFlight.current = null
          inFlightValue.current = null
        }
      },
      () => {
        if (inFlight.current === pending) {
          inFlight.current = null
          inFlightValue.current = null
        }
      },
    )
    return pending
  }

  const schedule = (next: string) => {
    if (timer.current !== null) window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => {
      void commit(next).catch(() => undefined)
    }, delay)
  }

  flushRef.current = () => (composing.current ? Promise.resolve() : commit(draftRef.current))

  return (
    <Textarea
      {...props}
      value={draft}
      onChange={(event) => {
        const next = event.currentTarget.value
        draftRef.current = next
        setDraft(next)
        if (!composing.current) schedule(next)
      }}
      onCompositionStart={() => {
        composing.current = true
      }}
      onCompositionEnd={(event) => {
        composing.current = false
        const next = event.currentTarget.value
        draftRef.current = next
        setDraft(next)
        schedule(next)
      }}
      onBlur={() => {
        void commit(draft).catch(() => undefined)
      }}
    />
  )
}
