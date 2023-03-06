/*using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using WebApplication1.Models;
using System.Globalization;

namespace WebApplication1.Controllers
{
    public class NotificationHelpClass : ControllerBase
    {

        public PaintingAnalyzerDbContext _context;

        public NotificationHelpClass(PaintingAnalyzerDbContext context)
        {
            _context = context;
        }
        public class AppJson
        {
            public string? Sender { get; set; }

            public string?[] Empfaenger { get; set; }

            public string content { get; set; }

            public string releasedate { get; set; }

            public bool Admins { get; set; }
        }
       
        public async Task<ActionResult<UserNotificationModel>> PostUserNotificationModel([FromBody] AppJson text)
        {
            if (_context.UserNotifications == null)
            {
                return Problem("Entity set 'PaintingAnalyzerDbContext.UserNotifications'  is null.");
            }
            NotificationModel notification = new NotificationModel()
            {
                NotificationID = 0,
                Content = text.content,
                ReleaseDate = DateTime.Parse(text.releasedate)
                
            };

            _context.Notifications.Add(notification);
            await _context.SaveChangesAsync();

            var notifyid = notification.NotificationID;

            Console.WriteLine("Die Id lautet: " + notifyid);

            if (text.Admins)
            {

            }


            foreach (string t in text.Empfaenger)
            {
                var User = _context.Users
                  .Where(b => b.UserName == t)
                  .FirstOrDefault();

                UserNotificationModel userNotification = new UserNotificationModel();
                {
                    userNotification.NotificationID = notifyid;
                    userNotification.UserID = User.UserID;

                }

            }

            Console.WriteLine(text.content);
            Console.WriteLine(text.Sender);
            foreach (string t in text.Empfaenger)
            {
                Console.WriteLine(text.content + " " + t);
            }
            return CreatedAtAction("GetUserNotificationModel", new { id = 0 }, text);
        }
    }
}


// GET: api/UserNotification/5
        [HttpGet("{id}")]
        public async Task<ActionResult<UserNotificationModel>> GetUserNotificationModel(int id)
        {
          if (_context.UserNotifications == null)
          {
              return NotFound();
          }
            var userNotificationModel = await _context.UserNotifications.FindAsync(id);

            if (userNotificationModel == null)
            {
                return NotFound();
            }

            return userNotificationModel;
        }




   /*

            _context.UserNotifications.Add(userNotificationModel);
            try
            {
                await _context.SaveChangesAsync();
            }
            catch (DbUpdateException)
            {
                if (UserNotificationModelExists(userNotificationModel.UserID))
                {
                    return Conflict();
                }
                else
                {
                    throw;
                }
            }
            
*
*
*
*
*
*
*
*
*
*
*public class AppJson
        {
            public string? Sender { get; set; }

            public string?[] Empfaenger { get; set; }

            public string content { get; set; }

            public string releasedate { get; set; }

            public bool Admins { get; set; }
        }
        [HttpPost("Postmethodetest")]
        public async Task<ActionResult<UserNotificationModel>> PostUserNotificationModel([FromBody] AppJson text)
        {
          if (_context.UserNotifications == null)
          {
              return Problem("Entity set 'PaintingAnalyzerDbContext.UserNotifications'  is null.");
          }
            NotificationModel notification = new NotificationModel()
            {
                NotificationID = 0,
                Content = text.content,
                ReleaseDate = DateTime.Parse(text.releasedate)
            };

            _context.Notifications.Add(notification);
            await _context.SaveChangesAsync();

            var notifyid = notification.NotificationID;

            Console.WriteLine("Die Id lautet: " + notifyid);
         
            if(text.Admins)
            {

            }

       
            foreach (string t in text.Empfaenger)
            {
                var User = _context.Users
                  .Where(b => b.UserName == t)
                  .FirstOrDefault();

                UserNotificationModel userNotification = new UserNotificationModel();
                {
                    userNotification.NotificationID = notifyid;
                    userNotification.UserID = User.UserID;

                }

            }
            
            Console.WriteLine(text.content);
            Console.WriteLine(text.Sender);
            foreach ( string t in text.Empfaenger)
            {
                Console.WriteLine(text.content + " " + t);
            }
            return CreatedAtAction("GetUserNotificationModel", new { id = 0 }, text);
        }
*/