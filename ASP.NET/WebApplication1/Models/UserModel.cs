using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace WebApplication1.Models
{
    public class UserModel
    {
        [Key]
        public int UserID { get; set; }

        public string Role { get; set; }

        public string ?UserName { get; set; }

        public string ?Email { get; set; }

        public byte[] PasswordHash { get; set; }

        public byte[] PasswordSalt { get; set; }

        [DataType(DataType.Date)]
        public DateTime ReleaseDate { get; set; }
        
        public virtual ICollection<NotificationModel> ?Notifications { get; set; } = new List<NotificationModel>();

        public virtual ICollection<ImageModel> ?UploadeImages { get; set; } = new List<ImageModel>();


    }
}
