from pydantic import BaseModel, Field


class SemanticChunkGroup(BaseModel):
    indices: list[int] = Field(
        description="Unit indices that belong to the same semantic chunk",
    )


class SemanticChunkGroups(BaseModel):
    groups: list[SemanticChunkGroup] = Field(
        description="Semantic groupings of unit indices",
    )
