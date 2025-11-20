package bo.edu.cba.faceid;

import androidx.room.ColumnInfo;
import androidx.room.Entity;
import androidx.room.PrimaryKey;

@Entity(tableName = "users")
public class User {
    @PrimaryKey(autoGenerate = true)
    public int id;

    @ColumnInfo(name = "name")
    public String name;

    @ColumnInfo(name = "face_embedding", typeAffinity = ColumnInfo.BLOB)
    public byte[] faceEmbedding;
}
