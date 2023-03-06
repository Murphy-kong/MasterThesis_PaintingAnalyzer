using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace WebApplication1.Models
{
    public class Session_Style_AccuracyModel
    {
        [Key]
        public int StyleID { get; set; }

        public string StyleName { get; set; }

        public float Accuracy { get; set; }

        public int? AnalyzeID { get; set; }

        [JsonIgnore]

        public AnalyzeModel? Analyze { get; set; }
    }
}
