use std::fmt;

use anyhow::Result;
use koharu_pipeline::{InpaintingModelChoice, PipelineConfig};
use koharu_renderer::TypesettingConfig;
use koharu_secrets::ExposeSecret as _;
use koharu_translator::{Language, Model, Provider, ProviderConfig, ProvidersConfig};
use serde::{Deserialize, Serialize};
use specta::Type;

use super::Error;

#[derive(Clone, Debug, Serialize, Type)]
pub struct Preferences {
    pub pipeline: PipelineConfig,
    pub providers: ProviderPreferences,
    pub typesetting: TypesettingConfig,
    pub languages: Vec<LanguageChoice>,
}

impl Preferences {
    pub(crate) fn load() -> Result<Self> {
        let pipeline = PipelineConfig::load()?;
        let providers = ProvidersConfig::load()?;
        let typesetting = TypesettingConfig::load()?;
        let pipeline = pipeline.read()?;
        let providers = providers.read()?;
        let typesetting = typesetting.read()?;
        Ok(Self {
            pipeline: pipeline.clone(),
            providers: ProviderPreferences::from_config(&providers)?,
            typesetting: typesetting.clone(),
            languages: Language::ALL
                .iter()
                .map(|language| LanguageChoice {
                    tag: language.tag().to_owned(),
                    name: language.to_string(),
                })
                .collect(),
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Type)]
pub struct ProviderPreferences {
    pub entries: Vec<ProviderPreference>,
    pub fal: CredentialInput,
}

#[derive(Clone, Debug, Deserialize, Serialize, Type)]
pub struct ProviderPreference {
    pub name: String,
    pub config: ProviderConfig,
    pub credential: Option<CredentialInput>,
}

impl ProviderPreferences {
    fn from_config(config: &ProvidersConfig) -> Result<Self> {
        let entries = config
            .entries()
            .into_iter()
            .map(|config| {
                let provider = config.provider();
                let credential = if provider == Provider::Local {
                    None
                } else {
                    let key: &'static str = provider.into();
                    Some(CredentialInput::load(key)?)
                };
                Ok(ProviderPreference {
                    name: provider.name().to_owned(),
                    config,
                    credential,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            entries,
            fal: CredentialInput::load("fal")?,
        })
    }

    fn into_config(self) -> Result<ProvidersConfig> {
        let mut configs = Vec::with_capacity(self.entries.len());
        let mut credentials = Vec::with_capacity(self.entries.len().saturating_sub(1));
        for entry in self.entries {
            let provider = entry.config.provider();
            match entry.credential {
                None if provider == Provider::Local => {}
                Some(credential) if provider != Provider::Local => {
                    let key: &'static str = provider.into();
                    credentials.push((key, credential));
                }
                None => anyhow::bail!("missing credential input for {provider}"),
                Some(_) => anyhow::bail!("local translation does not accept credentials"),
            }
            configs.push(entry.config);
        }
        let config = ProvidersConfig::from_entries(configs)?;
        for (key, credential) in credentials {
            credential.save(key)?;
        }
        self.fal.save("fal")?;
        Ok(config)
    }
}

#[derive(Clone, Default, Deserialize, Serialize, Type)]
pub struct CredentialInput {
    pub configured: bool,
    pub value: Option<String>,
    pub clear: bool,
}

impl fmt::Debug for CredentialInput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CredentialInput")
            .field("configured", &self.configured)
            .field("value", &self.value.as_ref().map(|_| "[REDACTED]"))
            .field("clear", &self.clear)
            .finish()
    }
}

impl CredentialInput {
    fn load(key: &str) -> Result<Self> {
        Ok(Self {
            configured: koharu_secrets::get(key)?
                .is_some_and(|secret| !secret.expose_secret().trim().is_empty()),
            value: None,
            clear: false,
        })
    }

    fn save(self, key: &str) -> Result<()> {
        if self.clear {
            koharu_secrets::delete(key)?;
        } else if let Some(value) = self.value {
            if value.trim().is_empty() {
                koharu_secrets::delete(key)?;
            } else {
                koharu_secrets::set(key, &value.into())?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct LanguageChoice {
    pub tag: String,
    pub name: String,
}

#[tracing::instrument(
    target = "koharu_metrics",
    name = "preferences_saved",
    skip_all,
    fields(setting = "application")
)]
#[tauri::command]
#[specta::specta]
pub(crate) async fn save_preferences(
    mut pipeline: PipelineConfig,
    providers: ProviderPreferences,
    typesetting: TypesettingConfig,
) -> std::result::Result<Preferences, Error> {
    pipeline.validate()?;
    remember_pipeline_profiles(&mut pipeline);
    let providers = providers.into_config()?;
    let pipeline_config = PipelineConfig::load()?;
    let providers_config = ProvidersConfig::load()?;
    let typesetting_config = TypesettingConfig::load()?;
    {
        let mut current = pipeline_config.write()?;
        *current = pipeline;
        current.save()?;
    }
    {
        let mut current = providers_config.write()?;
        *current = providers;
        current.save()?;
    }
    {
        let mut current = typesetting_config.write()?;
        *current = typesetting;
        current.save()?;
    }
    let preferences = Preferences::load()?;
    tracing::info!(
        target: "koharu_metrics",
        metric = "preference_changed",
        setting = "application",
    );
    Ok(preferences)
}

fn remember_pipeline_profiles(config: &mut PipelineConfig) {
    let koharu_pipeline::DetectionModel::KoharuLayoutRFDetrSeg2XL(settings) = &config.detection;
    config.processor.koharu_layout_rfdetr_seg_2xl = Some(settings.clone());
    if let koharu_pipeline::LocalInpaintingModel::Flux2Klein(settings) =
        &config.inpainting.local_model
    {
        config.processor.flux2_klein = Some(settings.clone());
    }
    if let koharu_pipeline::LocalInpaintingModel::RoremMixed(settings) =
        &config.inpainting.local_model
    {
        config.processor.rorem_mixed = Some(settings.clone());
    }
    if config.inpainting.api.provider == koharu_pipeline::InpaintingProvider::Fal {
        let settings = koharu_pipeline::ImageEditConfig {
            prompt: config.inpainting.api.prompt.clone(),
            apply_mode: config.inpainting.api.apply_mode,
        };
        match config.inpainting.api.model.as_str() {
            "microsoft/mai-image-2.5/edit" => config.processor.mai_image_2_5_edit = Some(settings),
            "microsoft/mai-image-2.5-pro/edit" => {
                config.processor.mai_image_2_5_pro_edit = Some(settings)
            }
            _ => {}
        }
    }
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn get_preferences() -> std::result::Result<Preferences, Error> {
    Ok(Preferences::load()?)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn get_translation_models() -> std::result::Result<Vec<Model>, Error> {
    Ok(koharu_translator::Translator::models().await?)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn get_inpainting_models() -> std::result::Result<Vec<InpaintingModelChoice>, Error>
{
    Ok(koharu_pipeline::inpainting_models().await?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remembers_fal_edit_profiles_independently() {
        let mut config = PipelineConfig::default();
        config.inpainting.method = koharu_pipeline::InpaintingMethod::Api;
        config.inpainting.api.provider = koharu_pipeline::InpaintingProvider::Fal;
        config.inpainting.api.model = "microsoft/mai-image-2.5/edit".to_owned();
        config.inpainting.api.prompt = "Clean the page.".to_owned();
        config.inpainting.api.apply_mode = koharu_pipeline::InpaintingApplyMode::Mask;
        remember_pipeline_profiles(&mut config);

        config.inpainting.api.model = "microsoft/mai-image-2.5-pro/edit".to_owned();
        config.inpainting.api.prompt = "Rebuild the page.".to_owned();
        config.inpainting.api.apply_mode = koharu_pipeline::InpaintingApplyMode::FullPage;
        remember_pipeline_profiles(&mut config);

        let edit = config.processor.mai_image_2_5_edit.unwrap();
        assert_eq!(edit.prompt, "Clean the page.");
        assert_eq!(edit.apply_mode, koharu_pipeline::InpaintingApplyMode::Mask);
        let pro = config.processor.mai_image_2_5_pro_edit.unwrap();
        assert_eq!(pro.prompt, "Rebuild the page.");
        assert_eq!(
            pro.apply_mode,
            koharu_pipeline::InpaintingApplyMode::FullPage
        );
    }
}
