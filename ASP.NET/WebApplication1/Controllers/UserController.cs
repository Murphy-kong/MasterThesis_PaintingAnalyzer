using System;
using System.Collections.Generic;
using System.IdentityModel.Tokens.Jwt;
using System.Linq;
using System.Security.Claims;
using System.Security.Cryptography;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.IdentityModel.Tokens;
using WebApplication1.Models;

namespace WebApplication1.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class UserController : ControllerBase
    {
        private readonly PaintingAnalyzerDbContext _context;
        private readonly IWebHostEnvironment _hostEnvironment;
        private readonly IConfiguration _configuration;

        public UserController(PaintingAnalyzerDbContext context, IWebHostEnvironment hostEnvironment, IConfiguration configuration)
        {
            _context = context;
            this._hostEnvironment = hostEnvironment;
            _configuration = configuration;
        }

        // GET: api/User
        [HttpGet, Authorize(Roles = "Admin")]
        public async Task<ActionResult<IEnumerable<UserModel>>> GetUsers()
        {
            Console.WriteLine(_hostEnvironment.ContentRootPath);
            if (_context.Users == null)
            {
                return NotFound();
            }
            return await _context.Users.ToListAsync();
        }

        [HttpGet("GetAllUserswithAvatars") ]
        public async Task<ActionResult<IEnumerable<UserModel>>> GetUsersWithNotifications()
        {
            var user = await _context.Users
                .Include(c => c.UploadeImages).ThenInclude(s => s.Analyzes)
                .ToListAsync();

            foreach(var t in user)
            {
                if(t.UploadeImages.Count > 0)
                {
                    foreach (var i in t.UploadeImages)
                    {
                        i.ImageSrc = String.Format("{0}://{1}{2}/Images/{3}", Request.Scheme, Request.Host, Request.PathBase, i.ImageName);
                    }
                } else
                {
                    ImageModel defaultavatar = new ImageModel{
                        Occupation = "avatar",
                        ImageSrc = String.Format("{0}://{1}{2}/Images/{3}", Request.Scheme, Request.Host, Request.PathBase, "defaultavatar.jpg")
                    };
                    t.UploadeImages.Add(defaultavatar);
                }
                
            }

            if (user == null)
                return NotFound();
 
            return user;
        }

        // GET: api/User
        [HttpGet("GetMyself"), Authorize(Roles = "Admin,Registered User")]
        public async Task<ActionResult<UserModel>> GetMyself()
        {

            var userName = User?.Identity?.Name;
            var user = await _context.Users
                .Where(c => c.UserName == User.FindFirstValue(ClaimTypes.Name))
                .FirstOrDefaultAsync();


            if (EqualityComparer<UserModel>.Default.Equals(user, default(UserModel)))
                return BadRequest(" Username not found.");

            return Ok(user);
        }

        // GET: api/User/5
        [HttpGet("{id}")]
        public async Task<ActionResult<UserModel>> GetUserModel(int id)
        {
            if (_context.Users == null)
            {
                return NotFound();
            }
            var user = await _context.Users
                .Where(c => c.UserID == id)
                .FirstOrDefaultAsync();


            if (user == null)
            {
                return NotFound();
            }
            return user;
        }


        [HttpGet("GetUsersWithNotifications")]
        public async Task<ActionResult<UserModel>> GetUsersWithNotifications(int id_user)
        {
            var user = await _context.Users
                .Where(c => c.UserID == id_user)
                .Include(c => c.Notifications)
                .FirstOrDefaultAsync();

            if (user == null)
                return NotFound();

            return user;
        }


        [HttpGet("GetMyselfWithNotifications"), Authorize]
        public async Task<ActionResult<UserModel>> GetMyselfWithNotifications()
        {
            var user = await _context.Users
                .Where(c => c.UserName == User.FindFirstValue(ClaimTypes.Name))
                .Include(c => c.Notifications)
                .FirstOrDefaultAsync();


            if (EqualityComparer<UserModel>.Default.Equals(user, default(UserModel)))
                return BadRequest(" Username not found.");

            return user;
        }

        [HttpGet("GetImagesfromUser"), Authorize]
        public async Task<ActionResult<IEnumerable<ImageModel>>> GetUsersWithUploadImages(int id_user)
        {
            var images = await _context.Images
                .Where(c => c.UserID == id_user)
                .ToListAsync();

            if (images == null)
                return NotFound();

            return images;
        }

        [HttpGet("GetImagesfromUserwithClaim"), Authorize(Roles = "Admin,Registered User")]
        public async Task<ActionResult<IEnumerable<ImagetGetDto>>> GetUsersWithUploadImages()
        {
 
            var images = await _context.Images
                    .Select(x => new ImagetGetDto()
                    {
                        ImageID = x.ImageID,
                        UserID = x.UserID,
                        userName = x.User.UserName,
                        Occupation = x.Occupation,
                        ImageName = x.ImageName,
                        ImageSrc = String.Format("{0}://{1}{2}/Images/{3}", Request.Scheme, Request.Host, Request.PathBase, x.ImageName)
                    })
                .Where(c => c.UserID.ToString() == User.FindFirstValue(ClaimTypes.NameIdentifier))
                .ToListAsync();

            if (images == null)
                return NotFound();

            return images;
        }

        [HttpGet("GetAvatarfromUser"), Authorize]
        public async Task<ActionResult<ImageModel>> GetUsersWithAvatarImage(int id_user)
        {
            var image = await _context.Images
                .Where(c => c.UserID == id_user && c.Occupation == "Avatar")
                .FirstOrDefaultAsync();

            if (image == null)
                return NotFound();

            return image;
        }

        [HttpGet("GetAvatarfromCurrentUser"), Authorize]
        public async Task<ActionResult<ImageModel>> GetUsersWithAvatarImageFromCurrent()
        {
            var userName = User?.Identity?.Name;
            var user = await _context.Users
                .Where(c => c.UserName == User.FindFirstValue(ClaimTypes.Name))
                .FirstOrDefaultAsync();

            var image = await _context.Images
                .Where(c => c.UserID == user.UserID && c.Occupation == "Avatar")
                .FirstOrDefaultAsync();

            image.ImageSrc = String.Format("{0}://{1}{2}/Images/{3}", Request.Scheme, Request.Host, Request.PathBase,image.ImageName);

            if (image == null)
                return NotFound();

            return image;
        }

        [HttpGet("GetAnalyzesfromUser")]
        public async Task<ActionResult<IEnumerable<UserModel>>> GetUsersWithAnalyzes(int id_user)
        {
            var analyzes = await _context.Users
               .Include(i => i.UploadeImages).ThenInclude(s => s.Analyzes)
               .Where(i => i.UploadeImages.Any(q => q.Analyzes.Any()) && i.UserID == id_user)
               .ToListAsync();

            /*  var analyzes = await _context.UploadImages
                   .Include(s => s.Analyzes)
                   .Where(i => i.Analyzes.Any() && i.UserID == id_user)
                   .ToListAsync();*/

            if (analyzes == null)
                return NotFound();

            return analyzes;
        }

        // PUT: api/User/5
        // To protect from overposting attacks, see https://go.microsoft.com/fwlink/?linkid=2123754
        [HttpPut("ChangeUserasAdmin")]
        public async Task<IActionResult> Put_ChangeUserAsAdmin(int id, UserDto request)
        {
       
            var userModel = await _context.Users
                .Where(c => c.UserName == request.Username)
                .FirstOrDefaultAsync();

            if (id != userModel.UserID)
            {
                return BadRequest();
            }

            CreatePasswordHash(request.Password, out byte[] passwordHash, out byte[] passwordSalt);
            userModel.UserName = request.Username;
            userModel.Email = request.Email;
            userModel.PasswordHash = passwordHash;  
            userModel.PasswordSalt= passwordSalt;

           
            _context.Entry(userModel).State = EntityState.Modified;

            try
            {
                await _context.SaveChangesAsync();
            }
            catch (DbUpdateConcurrencyException)
            {
                if (!UserModelExists(id))
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

        [HttpPut("ChangeUserasMyself"), Authorize(Roles = "Admin,Registered User")]
        public async Task<IActionResult> Put_ChangeUserAsMyself(UserChangeDto request)
        {

            var userModel = await _context.Users
                 .Where(c => c.UserName == User.FindFirstValue(ClaimTypes.Name))
                 .FirstOrDefaultAsync();

            if (EqualityComparer<UserModel>.Default.Equals(userModel, default(UserModel)))
                return BadRequest(" Username not found.");

            if (!VerifyPasswordHash(request.OldPassword, userModel))
                return BadRequest("Password is wrong.");

            if (request.Email != "")
                userModel.Email = request.Email;

            if(request.NewPassword != "")
            {
                CreatePasswordHash(request.NewPassword, out byte[] passwordHash, out byte[] passwordSalt);
                userModel.PasswordHash = passwordHash;
                userModel.PasswordSalt = passwordSalt;
            }
            


            _context.Entry(userModel).State = EntityState.Modified;

            try
            {
                await _context.SaveChangesAsync();
            }
            catch (DbUpdateConcurrencyException)
            {
                if (!UserModelExists(userModel.UserID))
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



        // POST: api/User
        // To protect from overposting attacks, see https://go.microsoft.com/fwlink/?linkid=2123754
        [HttpPost("AddRegisteredUser")]
        public async Task<ActionResult<UserModel>> PostUserModel_1(UserDto request)
        {
            var user = await _context.Users
                .Where(c => c.UserName == request.Username)
                .FirstOrDefaultAsync();

            if (!EqualityComparer<UserModel>.Default.Equals(user, default(UserModel)))
                return BadRequest(" Username already taken.");

            CreatePasswordHash(request.Password, out byte[] passwordHash, out byte[] passwordSalt);
            
            UserModel newuser = new UserModel();
            newuser.Role = "Registered User";
            newuser.UserName = request.Username;
            newuser.Email = request.Email;
            newuser.PasswordHash = passwordHash;    
            newuser.PasswordSalt = passwordSalt;

            if (_context.Users == null)
            {
                return Problem("Entity set 'PaintingAnalyzerDbContext.Users'  is null.");
            }
            _context.Users.Add(newuser);
            await _context.SaveChangesAsync();

            return CreatedAtAction("GetUserModel", new { id = newuser.UserID }, newuser);
        }

        [HttpPost("AddAdminUser"), Authorize(Roles = "Admin")]
        public async Task<ActionResult<UserModel>> PostUserModel_2(UserDto request)
        {
            var user = await _context.Users
                .Where(c => c.UserName == request.Username)
                .FirstOrDefaultAsync();

            if (!EqualityComparer<UserModel>.Default.Equals(user, default(UserModel)))
                return BadRequest(" Username already taken.");

            CreatePasswordHash(request.Password, out byte[] passwordHash, out byte[] passwordSalt);

            UserModel newuser = new UserModel();
            newuser.Role = "Admin";
            newuser.UserName = request.Username;
            newuser.Email = request.Email;
            newuser.PasswordHash = passwordHash;
            newuser.PasswordSalt = passwordSalt;

            if (_context.Users == null)
            {
                return Problem("Entity set 'PaintingAnalyzerDbContext.Users'  is null.");
            }
            _context.Users.Add(newuser);
            await _context.SaveChangesAsync();

            return CreatedAtAction("GetUserModel", new { id = newuser.UserID }, newuser);
        }

        
        [NonAction]
        public async Task<string> ImagepathCreator(string original_name)
        {
            string testimagename = original_name;
            string imageName = new String(Path.GetFileNameWithoutExtension(testimagename));
            imageName = imageName + Path.GetExtension(testimagename);
            var imagePath = Path.Combine(_hostEnvironment.ContentRootPath, "Images", imageName);
            return imagePath;
        }

        [HttpPost("AddDummyData")]
        public async Task<ActionResult<UserModel>> AddDummyData()
        {
            

            UserModel newuser = new UserModel { UserName = "Eddy" , Role = "Admin", Email = "eddy.khalili.sabet@gmail.com"};
            CreatePasswordHash("Test1", out byte[] passwordHash, out byte[] passwordSalt);
            newuser.PasswordHash = passwordHash;
            newuser.PasswordSalt = passwordSalt;

            UserModel newuser2 = new UserModel { UserName = "Daniela", Role = "Admin", Email = "beateverfuege@gmail.com" };
            CreatePasswordHash("Test2", out passwordHash, out passwordSalt);
            newuser2.PasswordHash = passwordHash;
            newuser2.PasswordSalt = passwordSalt;

            UserModel newuser3 = new UserModel { UserName = "Markus", Role = "Registered User", Email = "testmail1t@gmail.com" };
            CreatePasswordHash("Test3", out passwordHash, out passwordSalt);
            newuser3.PasswordHash = passwordHash;
            newuser3.PasswordSalt = passwordSalt;

            UserModel newuser4 = new UserModel { UserName = "Dennis", Role = "Registered User", Email = "testmail1@gmail.com" };
            CreatePasswordHash("Test4", out passwordHash, out passwordSalt);
            newuser4.PasswordHash = passwordHash;
            newuser4.PasswordSalt = passwordSalt;

            string[] imagenames = 
            { "1665_Girl_225744491.jpg", "FA10105_22223324095.jpg", "P1280937-b.jpg", "GiovanniB223300622.jpg", "Pieter_Pau225951767.jpg",
              "avatar.jpg", "Scuola_Gra225313192.jpg", "van-gogh-s225813852.jpg", "avatar2.jpg"
            };

            if (_context.Users == null)
            {
                return Problem("Entity set 'PaintingAnalyzerDbContext.Users'  is null.");
            }
            _context.Users.Add(newuser);
            _context.Users.Add(newuser2);
            _context.Users.Add(newuser3);
            _context.Users.Add(newuser4);

             _context.Database.ExecuteSqlRaw("INSERT INTO dbo.Notifications(Content, ReleaseDate, Type ) VALUES('Markus hat sich registriert.', '2022-06-21T18:10:00', 'Register');");
             _context.Database.ExecuteSqlRaw("INSERT INTO dbo.Notifications(Content, ReleaseDate, Type) VALUES('Daniela hat sich registriert.', '2022-06-21T18:10:00', 'Register'); ");
             _context.Database.ExecuteSqlRaw("INSERT INTO dbo.Notifications(Content, ReleaseDate, Type) VALUES('Eddy hat sich registriert.', '2022-06-21T18:10:00', 'Register');");
             await _context.SaveChangesAsync();

             _context.Database.ExecuteSqlRaw("INSERT INTO dbo.NotificationModelUserModel(NotificationsNotificationID, UsersUserID) VALUES(1, 1);");
             _context.Database.ExecuteSqlRaw("INSERT INTO dbo.NotificationModelUserModel(NotificationsNotificationID, UsersUserID) VALUES(2, 1); ");
             _context.Database.ExecuteSqlRaw("INSERT INTO dbo.NotificationModelUserModel(NotificationsNotificationID, UsersUserID) VALUES(1, 2); ");
             _context.Database.ExecuteSqlRaw("INSERT INTO dbo.NotificationModelUserModel(NotificationsNotificationID, UsersUserID) VALUES(2, 2); ");
             _context.Database.ExecuteSqlRaw("INSERT INTO dbo.NotificationModelUserModel(NotificationsNotificationID, UsersUserID) VALUES(1, 3);");
             _context.Database.ExecuteSqlRaw("INSERT INTO dbo.NotificationModelUserModel(NotificationsNotificationID, UsersUserID) VALUES(2, 3);");
             _context.Database.ExecuteSqlRaw("INSERT INTO dbo.Images(UserID, Occupation, ImageName, ReleaseDate) VALUES(1, 'Analyze', {0}, '2022-06-21T18:10:00'); ", imagenames[0]);
             _context.Database.ExecuteSqlRaw("INSERT INTO dbo.Images(UserID, Occupation, ImageName, ReleaseDate) VALUES(1, 'Analyze', {0}, '2022-06-21T18:10:00'); ", imagenames[1]);
            _context.Database.ExecuteSqlRaw("INSERT INTO dbo.Images(UserID, Occupation, ImageName, ReleaseDate) VALUES(1, 'Avatar', {0}, '2022-06-21T18:10:00'); ", imagenames[2]);
            _context.Database.ExecuteSqlRaw("INSERT INTO dbo.Images(UserID, Occupation, ImageName, ReleaseDate) VALUES(2, 'Analyze', {0}, '2022-06-21T18:10:00'); ", imagenames[3]);
             _context.Database.ExecuteSqlRaw(" INSERT INTO dbo.Images(UserID, Occupation, ImageName, ReleaseDate) VALUES(2, 'Analyze', {0}, '2022-06-21T18:10:00'); ", imagenames[4]);
            _context.Database.ExecuteSqlRaw("INSERT INTO dbo.Images(UserID, Occupation, ImageName, ReleaseDate) VALUES(2, 'Avatar', {0}, '2022-06-21T18:10:00'); ", imagenames[5]);
            _context.Database.ExecuteSqlRaw(" INSERT INTO dbo.Images(UserID, Occupation, ImageName, ReleaseDate) VALUES(3, 'Analyze', {0}, '2022-06-21T18:10:00'); ", imagenames[6]);
             _context.Database.ExecuteSqlRaw("INSERT INTO dbo.Images(UserID, Occupation, ImageName, ReleaseDate) VALUES(3, 'Analyze', {0}, '2022-06-21T18:10:00'); ; ", imagenames[7]);
            _context.Database.ExecuteSqlRaw(" INSERT INTO dbo.Images(UserID, Occupation, ImageName, ReleaseDate) VALUES(3, 'Avatar', {0},  '2022-06-21T18:10:00'); ", imagenames[8]);
            await _context.SaveChangesAsync();
             _context.Database.ExecuteSqlRaw("INSERT INTO dbo.Analyzes(ReleaseDate, ImageID) VALUES('2022-06-21T18:10:00', 1); ");
             _context.SaveChanges();
             _context.Database.ExecuteSqlRaw("INSERT INTO dbo.Session_Artist_Accuracy(ArtistName, Accuracy, AnalyzeID) VALUES('Johannes Vermeer', 70.5, 1); ");
             _context.Database.ExecuteSqlRaw("INSERT INTO dbo.Session_Genre_Accuracy(GenreName, Accuracy, AnalyzeID) VALUES('Renaissance', 75.2, 1); ");
             _context.Database.ExecuteSqlRaw("INSERT INTO dbo.Session_Style_Accuracy(StyleName, Accuracy, AnalyzeID) VALUES('Portrait', 78.3, 1); ");
             _context.Database.ExecuteSqlRaw("INSERT INTO dbo.Analyzes(ReleaseDate, ImageID) VALUES('2022-06-21T18:10:00', 3); ");
             await _context.SaveChangesAsync();
             _context.Database.ExecuteSqlRaw("INSERT INTO dbo.Session_Artist_Accuracy(ArtistName, Accuracy, AnalyzeID) VALUES('Giovanni Battista Tiepolo', 70.5, 2); ");
             _context.Database.ExecuteSqlRaw("INSERT INTO dbo.Session_Genre_Accuracy(GenreName, Accuracy, AnalyzeID) VALUES('Renaissance', 75.2, 2); ");
             _context.Database.ExecuteSqlRaw("INSERT INTO dbo.Session_Style_Accuracy(StyleName, Accuracy, AnalyzeID) VALUES('Illustration', 78.3, 2); ");
             _context.Database.ExecuteSqlRaw("INSERT INTO dbo.Analyzes(ReleaseDate, ImageID) VALUES('2022-06-21T18:10:00', 5); ");
             await _context.SaveChangesAsync();
             _context.Database.ExecuteSqlRaw("INSERT INTO dbo.Session_Artist_Accuracy(ArtistName, Accuracy, AnalyzeID) VALUES('Giovanni Battista Tiepolo', 70.5, 2); ");
             _context.Database.ExecuteSqlRaw("INSERT INTO dbo.Session_Genre_Accuracy(GenreName, Accuracy, AnalyzeID) VALUES('Renaissance', 75.2, 2); ");
             _context.Database.ExecuteSqlRaw("INSERT INTO dbo.Session_Style_Accuracy(StyleName, Accuracy, AnalyzeID) VALUES('Illustration', 78.3, 2); ");




            return CreatedAtAction("GetUserModel", new { id = newuser.UserID }, newuser);
        }


        [HttpGet("testDB")]
        public async Task<string> testDB()
        {
            Console.WriteLine(User?.Identity?.Name);
            Console.WriteLine(User.Claims);
            Console.WriteLine("Alles: " + Request.Scheme, Request.Host, Request.PathBase);
            Console.WriteLine("Request.Scheme: "+  Request.Scheme);
            Console.WriteLine("Request.Host: " + Request.Host);
            Console.WriteLine("Request.PathBase: " + Request.PathBase);

            return "test";
        }






        //Create a hashed Password
        private void CreatePasswordHash(string password, out byte[] passwordHash, out byte[] passwordSalt)
        {
            using (var hmac = new HMACSHA512())
            {
                passwordSalt = hmac.Key;
                passwordHash = hmac.ComputeHash(System.Text.Encoding.UTF8.GetBytes(password));
                Console.WriteLine(passwordSalt);
                Console.WriteLine(passwordHash);
            }
        }


        //Create a hashed Password
        private bool VerifyPasswordHash(string password, UserModel User)
        {
            using (var hmac = new HMACSHA512(User.PasswordSalt))
            {
              
                var computedHash = hmac.ComputeHash(System.Text.Encoding.UTF8.GetBytes(password));
                return computedHash.SequenceEqual(User.PasswordHash);
            }
        }

        [HttpPost("Login")]
        public async Task<ActionResult<UserModel>> Login(UserLoginDto request)
        {
            var user = await _context.Users
                .Where(c => c.UserName == request.Username)
                .FirstOrDefaultAsync();

            if (EqualityComparer<UserModel>.Default.Equals(user, default(UserModel)))
                return BadRequest(" User couldnt be found.");

            if (!VerifyPasswordHash(request.Password, user ))
                return BadRequest("Password is wrong.");

            string token = CreateToken(user);
            

  
            return Ok(token);
        }




        private string CreateToken(UserModel user)
        {
            List<Claim> claims = null;
            if (user.Role == "Admin")
            {
                List<Claim> claimstmp = new List<Claim>
                {
                    new Claim(ClaimTypes.Name, user.UserName),
                    new Claim(ClaimTypes.Role, user.Role),
                    new Claim(ClaimTypes.NameIdentifier as string , user.UserID.ToString())
                };
                claims = claimstmp;
            }

            if (user.Role == "Registered User")
            {
                List<Claim> claimstmp = new List<Claim>
                {
                    new Claim(ClaimTypes.Name, user.UserName),
                    new Claim(ClaimTypes.Role, user.Role),
                    new Claim(ClaimTypes.NameIdentifier as string , user.UserID.ToString())
                };
                claims = claimstmp;
            }



            var key = new SymmetricSecurityKey(System.Text.Encoding.UTF8.GetBytes(
                _configuration.GetSection("AppSettings:Token").Value));

            var cred = new SigningCredentials(key, SecurityAlgorithms.HmacSha512Signature);

            var token = new JwtSecurityToken(
                claims: claims,
                expires: DateTime.Now.AddMinutes(30),
                signingCredentials: cred);
         
            var jwt = new JwtSecurityTokenHandler().WriteToken(token);

            return jwt;
        }

        [HttpPost("AddUserNotification")]
        public async Task<ActionResult<UserModel>> PostUserNotificationModel(int id_user, int id_notification)
        {
            var user = await _context.Users
                .Where(c => c.UserID == id_user)
                .Include(c => c.Notifications)
                .FirstOrDefaultAsync();

            if (user == null)
                return NotFound();

            var notification = await _context.Notifications.FindAsync(id_notification);
            if (notification == null)
                return NotFound();

            user.Notifications.Add(notification);
            await _context.SaveChangesAsync();
            return user;
        }

        [HttpPost("AddUserNotificationtoallAdmins")]
        public async Task<ActionResult<NotificationModel>> PostUserNotificationModelToAllAdmins(int id_notification)
        {

            var user = await _context.Users
                .Where(c => c.Role == "Admin")
                .Include(c => c.Notifications)
                .ToListAsync();

            if (user == null)
                return NotFound();

            var notification = await _context.Notifications.FindAsync(id_notification);
            if (notification == null)
                return NotFound();

            foreach (var element in user)
            {
                element.Notifications.Add(notification);
                Console.WriteLine(element.UserName);
            }

            await _context.SaveChangesAsync();
            return Ok(notification);
        }


        // DELETE: api/User/5
        [HttpDelete("{id}"), Authorize(Roles = "Admin")]
        public async Task<IActionResult> DeleteUserModel(int id)
        {
            if (_context.Users == null)
            {
                return NotFound();
            }
            var userModel = await _context.Users.FindAsync(id);
            if (userModel == null)
            {
                return NotFound();
            }

            _context.Users.Remove(userModel);
            await _context.SaveChangesAsync();

            return NoContent();
        }

        // DELETE: api/User/5
        [HttpDelete("DeleteTables")]
        public async Task<IActionResult> DeleteAll()
        {
            _context.Database.ExecuteSqlRaw("DROP TABLE dbo.Session_Artist_Accuracy");
            _context.Database.ExecuteSqlRaw("DROP TABLE dbo.Session_Genre_Accuracy");
            _context.Database.ExecuteSqlRaw("DROP TABLE dbo.Session_Style_Accuracy");
            _context.Database.ExecuteSqlRaw("DROP TABLE dbo.Analyzes");
            _context.Database.ExecuteSqlRaw("DROP TABLE dbo.NotificationModelUserModel");
            _context.Database.ExecuteSqlRaw("DROP TABLE dbo.Images");
            _context.Database.ExecuteSqlRaw("DROP TABLE dbo.Notifications");
            _context.Database.ExecuteSqlRaw("DROP TABLE dbo.Users");
            await _context.SaveChangesAsync();

            return NoContent();
        }

    

        private bool UserModelExists(int id)
        {
            return (_context.Users?.Any(e => e.UserID == id)).GetValueOrDefault();
        }
    }
}
