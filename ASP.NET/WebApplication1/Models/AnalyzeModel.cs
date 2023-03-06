using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace WebApplication1.Models
{
    public class AnalyzeModel
    {
        [Key]
        public int? AnalyzeID { get; set; }

        public int? ImageID { get; set; }

        [JsonIgnore]
  
        public ImageModel? Image { get; set; }
        

        public string? ML_Model { get; set; }

        public ICollection<Session_Artist_AccuracyModel> Session_Artist_Accuracy { get; set; } = new List<Session_Artist_AccuracyModel>();

        public ICollection<Session_Genre_AccuracyModel> Session_Genre_Accuracy { get; set; } = new List<Session_Genre_AccuracyModel>();

        public ICollection<Session_Style_AccuracyModel> Session_Style_Accuracy { get; set; } = new List<Session_Style_AccuracyModel>();

        [DataType(DataType.Date)]
        public DateTime ReleaseDate { get; set; }

    }
}
