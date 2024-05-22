from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    PROJECT_NAME: str = "Owlracer Pipeline"

    MLFLOW_S3_ENDPOINT_URL: str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    # MLFLOW_PROJECT_ENV: str
    MLFLOW_S3_BUCKET: str | None = None
    REMOTE_SERVER_URI: str

settings = Settings()