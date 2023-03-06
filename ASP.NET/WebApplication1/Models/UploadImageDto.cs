namespace WebApplication1.Models
{
    public class UploadImageDto
    {
        public int? UploadImageID { get; set; }

        public int? UserID { get; set; }

        public string Occupation { get; set; }

        public string? ImageName { get; set; }

        public IFormFile ImageFile { get; set; }
 
        public string? ImageSrc { get; set; }

        public DateTime ReleaseDate { get; set; }

    }
}
