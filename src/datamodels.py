from sqlmodel import Field, SQLModel

class Data(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    path: str
    x_pos: float
    y_pos: float