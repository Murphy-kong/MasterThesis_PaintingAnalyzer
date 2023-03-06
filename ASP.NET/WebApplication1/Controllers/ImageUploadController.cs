#nullable disable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using WebApplication1.Models;
using System.Text.RegularExpressions;
using Microsoft.AspNetCore.Authorization;
using System.Security.Claims;


namespace WebApplication1.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ImageUploadController : ControllerBase
    {
        private readonly PaintingAnalyzerDbContext _context;
        private readonly IWebHostEnvironment _hostEnvironment;

        public ImageUploadController(PaintingAnalyzerDbContext context, IWebHostEnvironment hostEnvironment)
        {
            _context = context;
            this._hostEnvironment = hostEnvironment;
        }

        // GET: api/Employee
        [HttpGet]
        public async Task<ActionResult<IEnumerable<ImagetGetDto>>> GetUploadImages()
        {
            Console.WriteLine("GET METHODE");
            Console.WriteLine(String.Format("{0}://{1}{2}/Images/{3}", Request.Scheme, Request.Host, Request.PathBase, "Bildname"));
            var result =  await _context.Images
                .Include(x => x.User)
                .Select(x=> new ImagetGetDto()
                {
                ImageID = x.ImageID,
                UserID = x.UserID,
                userName = x.User.UserName,
                Occupation = x.Occupation,
                ImageName = x.ImageName,
                ImageSrc = String.Format("{0}://{1}{2}/Images/{3}", Request.Scheme, Request.Host, Request.PathBase, x.ImageName)
                })
               .ToListAsync();

            return result;

            //.Where(x => x.Occupation == "Analyze")
        }

        // GET: api/UploadImage/5
        [HttpGet("{id}")]
        public async Task<ActionResult<ImageModel>> GetUploadImageModel(int id)
        {
            var uploadImageModel = await _context.Images.FindAsync(id);

            if (uploadImageModel == null)
            {
                return NotFound();
            }

            return uploadImageModel;
        }

        [HttpGet("GetAnalyzesfromUploadImage")]
        public async Task<ActionResult<IEnumerable<AnalyzeModel>>> GetAnalyzesfromUploadImage(int id_image)
        {
            var analyzes = await _context.Analyzes
                .Where(c => c.ImageID == id_image)
                .Include(c => c.Session_Artist_Accuracy)
                .Include(c => c.Session_Genre_Accuracy)
                .Include(c => c.Session_Style_Accuracy)
                .ToListAsync();

            if (analyzes == null)
                return NotFound();

       

            return analyzes;
        }

        [HttpGet("GetImageswithAnalyzes"), Authorize(Roles = "Admin,Registered User")]
        public async Task<ActionResult<IEnumerable<ImageModel>>> GetImageswithAnalyzes()
        {
            var analyzes = await _context.Images
                .Where(c => c.Occupation == "Analyze")
                .Include(c => c.Analyzes).ThenInclude(c => c.Session_Artist_Accuracy)
                .Include(c => c.Analyzes).ThenInclude(c => c.Session_Genre_Accuracy)
                .Include(c => c.Analyzes).ThenInclude(c => c.Session_Style_Accuracy)
                .ToListAsync();

            if (analyzes == null)
                return NotFound();

            int i = 0;
            foreach (var t in analyzes)
            {
                analyzes[i].ImageSrc = String.Format("{0}://{1}{2}/Images/{3}", Request.Scheme, Request.Host, Request.PathBase, t.ImageName);
                i++;
            }

            return analyzes;
        }

        [HttpGet("GetAnalyzesFromUser"), Authorize(Roles = "Admin,Registered User")]
        public async Task<ActionResult<IEnumerable<ImageModel>>> GetAnalyzesFromUser()
        {
            var analyzes = await _context.Images
                .Where(c => c.Occupation == "Analyze" && c.UserID.ToString() == User.FindFirstValue(ClaimTypes.NameIdentifier) && c.Analyzes.Count > 0)
                .Include(c => c.Analyzes).ThenInclude(c => c.Session_Artist_Accuracy)
                .Include(c => c.Analyzes).ThenInclude(c => c.Session_Genre_Accuracy)
                .Include(c => c.Analyzes).ThenInclude(c => c.Session_Style_Accuracy)
                .ToListAsync();

            if (analyzes == null)
                return NotFound();

            int i = 0;
            foreach (var t in analyzes)
            {
                analyzes[i].ImageSrc = String.Format("{0}://{1}{2}/Images/{3}", Request.Scheme, Request.Host, Request.PathBase, t.ImageName);
                i++;
            }

            return analyzes;
        }
        [NonAction]
        public static bool Contains<T>(IEnumerable<NotificationModel> source, T value)
        {
            /*foreach (var i in source)
            {
                var name  = i.Content;
                var output = Regex.Replace(name.Split()[0], @"[^0-9a-zA-Z\ ]+", "");

                if (Equals(i, value))
                    return true;
            }*/
            return false;
        }
        [NonAction]
        public bool test(IEnumerable<NotificationModel> source, string username)
        {
            foreach (var i in source)
            {
               /* var name = i.Content;
                var output = Regex.Replace(name.Split()[0], @"[^0-9a-zA-Z\ ]+", "");
               
                if (Equals(i, username))
                    return true;*/
            }

                return false;
        }

        [HttpPost("GetAvatarsfromUserNotificationList"), Authorize(Roles = "Admin,Registered User")]
        public async Task<ActionResult<IEnumerable<ImageModel>>> GetAnalyzesfromUploadImage(IEnumerable<NotificationModel> notification)
        {
            List<string> termsList = new List<string>();
            foreach (var i in notification)
            {
                var name = i.Content;
                var output = Regex.Replace(name.Split()[0], @"[^0-9a-zA-Z\ ]+", "");
                termsList.Add(output);
            }
            foreach (var month in termsList)
            {
                Console.WriteLine(month);
            }

            List<ImageModel> result = new List<ImageModel>();
            foreach (var month in termsList)
            {
                var avatars = await _context.Images
                 .Where(c => (month == c.User.UserName) && (c.Occupation == "avatar"))
                 .Select(x => new ImageModel()
                 {
                     ImageID = x.ImageID,
                     UserID = x.UserID,
                     Occupation = x.Occupation,
                     ImageName = x.ImageName,
                     ImageSrc = String.Format("{0}://{1}{2}/Images/{3}", Request.Scheme, Request.Host, Request.PathBase, x.ImageName)
                 })
                .FirstOrDefaultAsync();
                result.Add(avatars);
            }

                //Contains(notification ,c.User.UserName)
                if (result == null)
                return NotFound();

            return result;
        }

        // PUT: api/UploadImage/5
        // To protect from overposting attacks, see https://go.microsoft.com/fwlink/?linkid=2123754
        [HttpPut("{id}")]
        public async Task<IActionResult> PutUploadImageModel(int id, [FromForm] ImageModel uploadImageModel)
        {
            Console.WriteLine("PUT METHODE");
            Console.WriteLine("id: " + id);
            Console.WriteLine("uploadImageModel.UploadImageID: " + uploadImageModel.ImageID);
            if (id != uploadImageModel.ImageID)
            {
                return BadRequest();
            }
            
            Console.WriteLine(id);
            if(uploadImageModel.ImageFile != null)
            {
                DeleteImage(uploadImageModel.ImageName);
                uploadImageModel.ImageName = await SaveImage(uploadImageModel.ImageFile);
            }

            _context.Entry(uploadImageModel).State = EntityState.Modified;

            try
            {
                await _context.SaveChangesAsync();
            }
            catch (DbUpdateConcurrencyException)
            {
                if (!UploadImageModelExists(id))
                {
                    return NotFound();
                }
                else
                {
                    throw;
                }
            }

            return NoContent();
        }

        // POST: api/UploadImage
        // To protect from overposting attacks, see https://go.microsoft.com/fwlink/?linkid=2123754
        [HttpPost, Authorize(Roles = "Admin,Registered User")]
        public async Task<ActionResult<ImageModel>> PostUploadImageModel([FromForm] UploadImageDto uploadImageModelrequest)
        {
         
            var user = await _context.Users
                .Where(c => c.UserID.ToString() == User.FindFirstValue(ClaimTypes.NameIdentifier))
                .FirstOrDefaultAsync();
            if (user == null)
                return NotFound();

            ImageModel uploadImageModel = new ImageModel();
            uploadImageModel.ImageID = 0;
            uploadImageModel.UserID = int.Parse(User.FindFirstValue(ClaimTypes.NameIdentifier));
            uploadImageModel.Occupation = uploadImageModelrequest.Occupation;
            uploadImageModel.ReleaseDate = uploadImageModelrequest.ReleaseDate;
            uploadImageModel.ImageName = await SaveImage(uploadImageModelrequest.ImageFile);
            uploadImageModel.User = user;
            _context.Images.Add(uploadImageModel);
            Console.WriteLine("USER ID LAUTET: " + uploadImageModel.UserID);
            await _context.SaveChangesAsync();

          

            return await GetUploadImageModel(uploadImageModel.ImageID);
        }

        // DELETE: api/UploadImage/5
        [HttpDelete("{id}")]
        public async Task<IActionResult> DeleteUploadImageModel(int id)
        {
            var uploadImageModel = await _context.Images.FindAsync(id);
            if (uploadImageModel == null)
            {
                return NotFound();
            }


            DeleteImage(uploadImageModel.ImageName);
            _context.Images.Remove(uploadImageModel);
            await _context.SaveChangesAsync();

            return NoContent();
        }
        [NonAction]
        private bool UploadImageModelExists(int id)
        {
            return _context.Images.Any(e => e.ImageID == id);
        }

        [NonAction]
        public async Task<string> SaveImage(IFormFile imageFile)
        {
            string imageName = new String(Path.GetFileNameWithoutExtension(imageFile.FileName).Take(10).ToArray()).Replace(' ', '-');
            imageName = imageName + DateTime.Now.ToString("yymmssfff") + Path.GetExtension(imageFile.FileName);
            var imagePath = Path.Combine(_hostEnvironment.ContentRootPath, "Images", imageName);
            Console.WriteLine(imagePath);
            using (var fileStream = new FileStream(imagePath, FileMode.Create))
            {
                await imageFile.CopyToAsync(fileStream);
            }
            Console.WriteLine(imageName);
            return imageName;
        }

        [NonAction]
        public void DeleteImage(string imageName)
        {
            var imagePath = Path.Combine(_hostEnvironment.ContentRootPath, "Images", imageName);
            if (System.IO.File.Exists(imagePath))
                System.IO.File.Delete(imagePath);
        }

    }
}
