'use client'

import { ChevronDown } from 'lucide-react'
import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { ModelPicker } from '@/components/controls/ModelPicker'
import { GenerationPreferences } from '@/components/preferences/GenerationPreferences'
import {
  PreferencePage,
  PreferenceRow,
  PreferenceSection,
} from '@/components/preferences/PreferenceFields'
import { chapterPromptPresets } from '@/lib/chapterPresets'
import {
  modelKey,
  orderedLanguageChoices,
  providerName,
  translationProfileSelection,
} from '@/lib/translation'
import type {
  LanguageChoice,
  Model,
  ProviderPreference,
  TranslationConfig as TranslationSettings,
  TranslationProfile,
  TranslationUnitPolicy,
} from '@koharu/bridge/protocol'
import { Badge } from '@koharu/ui/components/badge'
import { Popover, PopoverContent, PopoverTrigger } from '@koharu/ui/components/popover'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@koharu/ui/components/select'
import { Tabs, TabsList, TabsTrigger } from '@koharu/ui/components/tabs'
import { Textarea } from '@koharu/ui/components/textarea'

type ProfileName = 'page' | 'chapter'

export function TranslationPreferences({
  value,
  modelChoices,
  providers,
  languages,
  onChange,
}: {
  value: TranslationSettings
  modelChoices: Model[]
  providers: ProviderPreference[]
  languages: LanguageChoice[]
  onChange: (value: TranslationSettings) => void
}) {
  const { t } = useTranslation()
  const [profileName, setProfileName] = useState<ProfileName>('page')
  const [modelOpen, setModelOpen] = useState(false)
  const profile = value[profileName]
  const updateProfile = (next: TranslationProfile) => onChange({ ...value, [profileName]: next })
  const selected =
    modelChoices.find((candidate) => modelKey(candidate) === modelKey(profile.model)) ?? null
  const current: Model = selected ?? {
    ...profile.model,
    model: profile.model.model ?? null,
    name: profile.model.model ?? providerName(providers, profile.model.provider),
    quantizations: [],
    vision: profile.model.vision ?? false,
    reasoning: profile.model.reasoning ?? false,
    reasoning_required: profile.model.reasoning_required ?? false,
  }
  const choices = selected ? modelChoices : [current, ...modelChoices]
  const quantizations = current.quantizations
  const unitPolicy =
    profile.unit_policy ?? (profileName === 'chapter' ? 'adaptive_v1' : 'page_only')
  const selectedPreset =
    profileName === 'chapter'
      ? (chapterPromptPresets.find((preset) => preset.instructions === profile.instructions)?.id ??
        'custom')
      : 'custom'
  const languageChoices = useMemo(() => orderedLanguageChoices(languages), [languages])
  return (
    <PreferencePage
      title={t('settings.translation.title')}
      description={t('settings.translation.description')}
    >
      <Tabs
        value={profileName}
        onValueChange={(next) => setProfileName(next as ProfileName)}
        className='gap-4'
      >
        <TabsList className='grid w-full grid-cols-2'>
          <TabsTrigger value='page'>{t('settings.translation.pageProfile')}</TabsTrigger>
          <TabsTrigger value='chapter'>{t('settings.translation.chapterProfile')}</TabsTrigger>
        </TabsList>
        <PreferenceSection
          title={t('settings.translation.model')}
          description={t('settings.translation.modelDescription')}
        >
          <PreferenceRow
            title={t('settings.translation.translationModel')}
            description={t('settings.translation.translationModelDescription')}
          >
            <Popover open={modelOpen} onOpenChange={setModelOpen}>
              <PopoverTrigger
                type='button'
                aria-label={t('settings.translation.translationModel')}
                className='flex h-9 w-full min-w-0 items-center justify-between gap-2 rounded-lg border border-input bg-transparent px-2.5 text-[11px] transition-colors outline-none hover:bg-foreground/[0.03] focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50'
              >
                <span className='min-w-0 flex-1 text-left'>
                  <ModelLabel model={current} providers={providers} />
                </span>
                <ChevronDown className='size-3.5 shrink-0 text-muted-foreground' />
              </PopoverTrigger>
              <PopoverContent
                align='start'
                sideOffset={4}
                className='w-(--anchor-width) min-w-64 gap-0 overflow-hidden rounded-xl border border-border/50 p-1 shadow-sm ring-0'
              >
                <ModelPicker
                  value={profile.model}
                  models={choices}
                  providers={providers}
                  onBack={() => setModelOpen(false)}
                  onSelect={(model) => {
                    updateProfile(translationProfileSelection(profile, model))
                    setModelOpen(false)
                  }}
                />
              </PopoverContent>
            </Popover>
          </PreferenceRow>
          {quantizations.length > 0 && (
            <PreferenceRow
              title={t('settings.translation.quantization')}
              description={t('settings.translation.quantizationDescription')}
            >
              <Select
                value={profile.model.quantization ?? ''}
                onValueChange={(quantization) =>
                  updateProfile({ ...profile, model: { ...profile.model, quantization } })
                }
              >
                <SelectTrigger
                  aria-label={t('settings.translation.modelQuantization')}
                  className='h-8 w-full text-[11px]'
                >
                  <SelectValue placeholder={t('settings.translation.selectQuantization')} />
                </SelectTrigger>
                <SelectContent>
                  {quantizations.map((quantization) => (
                    <SelectItem key={quantization.id} value={quantization.id}>
                      {quantization.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </PreferenceRow>
          )}
          <PreferenceRow
            title={t('settings.translation.unitPolicy')}
            description={t('settings.translation.unitPolicyDescription')}
          >
            <Select
              value={unitPolicy}
              onValueChange={(unit_policy) =>
                updateProfile({
                  ...profile,
                  unit_policy: unit_policy as TranslationUnitPolicy,
                })
              }
            >
              <SelectTrigger
                aria-label={t('settings.translation.unitPolicy')}
                className='h-8 w-full text-[11px]'
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value='page_only'>{t('settings.translation.pageOnly')}</SelectItem>
                <SelectItem value='adaptive_v1'>{t('settings.translation.adaptiveV1')}</SelectItem>
              </SelectContent>
            </Select>
          </PreferenceRow>
        </PreferenceSection>

        <GenerationPreferences
          value={profile.generation}
          reasoningRequired={current.reasoning_required}
          onChange={(generation) => updateProfile({ ...profile, generation })}
        />

        <PreferenceSection title={t('settings.translation.instructions')}>
          {profileName === 'chapter' && (
            <PreferenceRow
              title={t('settings.translation.promptPreset')}
              description={t('settings.translation.promptPresetDescription')}
            >
              <Select
                value={selectedPreset}
                onValueChange={(presetId) => {
                  const preset = chapterPromptPresets.find((candidate) => candidate.id === presetId)
                  if (preset) updateProfile({ ...profile, instructions: preset.instructions })
                }}
              >
                <SelectTrigger
                  aria-label={t('settings.translation.promptPreset')}
                  className='h-8 w-full text-[11px]'
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value='custom'>{t('settings.translation.presetCustom')}</SelectItem>
                  {chapterPromptPresets.map((preset) => (
                    <SelectItem key={preset.id} value={preset.id}>
                      {t(preset.labelKey)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </PreferenceRow>
          )}
          <PreferenceRow
            title={t('model.instructions')}
            description={t('settings.translation.instructionsDescription')}
            align='start'
          >
            <Textarea
              aria-label={t('settings.translation.instructionsLabel')}
              value={profile.instructions ?? ''}
              className='field-sizing-fixed min-h-24 resize-y overflow-y-auto text-[12px] leading-5'
              placeholder={t('settings.translation.instructionsPlaceholder')}
              onChange={(event) =>
                updateProfile({ ...profile, instructions: event.currentTarget.value || null })
              }
            />
          </PreferenceRow>
        </PreferenceSection>
      </Tabs>

      <PreferenceSection title={t('settings.translation.output')}>
        <PreferenceRow
          title={t('settings.translation.sourceLanguage')}
          description={t('settings.translation.sourceLanguageDescription')}
        >
          <Select
            value={value.source_language}
            items={Object.fromEntries(
              languageChoices.map((language) => [language.tag, language.name]),
            )}
            onValueChange={(source_language) =>
              source_language && onChange({ ...value, source_language })
            }
          >
            <SelectTrigger
              aria-label={t('settings.translation.sourceLanguage')}
              className='h-8 w-full text-[11px]'
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {languageChoices.map((language) => (
                <SelectItem key={language.tag} value={language.tag}>
                  {language.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </PreferenceRow>
        <PreferenceRow
          title={t('model.targetLanguage')}
          description={t('settings.translation.targetLanguageDescription')}
        >
          <Select
            value={value.target_language}
            items={Object.fromEntries(
              languageChoices.map((language) => [language.tag, language.name]),
            )}
            onValueChange={(target_language) =>
              target_language && onChange({ ...value, target_language })
            }
          >
            <SelectTrigger
              aria-label={t('model.targetLanguage')}
              className='h-8 w-full text-[11px]'
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {languageChoices.map((language) => (
                <SelectItem key={language.tag} value={language.tag}>
                  {language.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </PreferenceRow>
      </PreferenceSection>
    </PreferencePage>
  )
}

function ModelLabel({ model, providers }: { model: Model; providers: ProviderPreference[] }) {
  return (
    <span className='flex min-w-0 items-center gap-2'>
      <Badge variant='outline' className='shrink-0 px-1.5 py-0 text-[9px] font-medium'>
        {providerName(providers, model.provider)}
      </Badge>
      <span className='truncate'>{model.name}</span>
    </span>
  )
}
