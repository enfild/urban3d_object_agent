from pydantic import BaseSettings, Field


def get_env_variable(name: str, default=None):
    return Field(default, env=name)


class Settings(BaseSettings):
    # OpenAI LLM settings
    openai_api_key: str = Field("", env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4", env="OPENAI_MODEL")

    # Milvus configuration
    milvus_host: str = Field("localhost", env="MILVUS_HOST")
    milvus_port: int = Field(19530, env="MILVUS_PORT")

    # 3D point cloud encoder service
    encoder_url: str = Field("http://inner-test.env:8922/3dpointsencoder", env="ENCODER_URL")

    # Callback for new or updated 3D objects
    new_object_url: str = Field("http://inner-test.env:8922/new3dobject", env="NEW_OBJ_URL")

    # Dimension of the 3D embeddings
    vector_dim: int = Field(256, env="VECTOR_DIM")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Instantiate settings for application
settings = Settings()