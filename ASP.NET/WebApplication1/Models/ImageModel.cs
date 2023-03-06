using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace WebApplication1.Models
{
    public class ImageModel
    {
        [Key]
        public int ImageID { get; set; }

        public int? UserID { get; set; }
        [JsonIgnore]
        public UserModel User { get; set; }

        [Column(TypeName = "nvarchar(50)")]
        public string Occupation { get; set; }

        [Column(TypeName = "nvarchar(100)")]
        [Required(AllowEmptyStrings = true),
         DisplayFormat(ConvertEmptyStringToNull = false)]
        public string? ImageName {get; set; } 

        [NotMapped]
        public IFormFile ImageFile { get; set; }
        [Required(AllowEmptyStrings = true),
         DisplayFormat(ConvertEmptyStringToNull = false)]
        [NotMapped]
        public string ImageSrc { get; set; }

        [DataType(DataType.Date)]
        public DateTime ReleaseDate { get; set; }

        public virtual ICollection<AnalyzeModel>? Analyzes { get; set; } = new List<AnalyzeModel>();


        //Pascal(EmployeeName) -> Camel
        //Camel(employeeName) -> Pascal
    }
}
