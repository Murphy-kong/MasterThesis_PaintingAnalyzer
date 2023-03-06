using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace WebApplication1.Models
{
    public class NotificationModel
    {
        [Key]
        public int NotificationID { get; set; }

        public string Content { get; set; }
        
        [JsonIgnore]
        public virtual ICollection<UserModel>? Users { get; set; }

        [DataType(DataType.Date)]
        public DateTime ReleaseDate { get; set; }


        public string Type { get; set; }

    }
}
