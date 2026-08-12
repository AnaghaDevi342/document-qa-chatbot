from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    azure_openai_api_key: str
    azure_openai_endpoint: str

    azure_openai_chat_deployment: str
    azure_openai_embedding_deployment: str

    elasticsearch_url: str

    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60

    app_username: str
    app_password: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

settings = Settings()