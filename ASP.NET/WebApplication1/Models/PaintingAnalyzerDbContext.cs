using Microsoft.EntityFrameworkCore;
using WebApplication1.Models;

namespace WebApplication1.Models
{
    public class PaintingAnalyzerDbContext: DbContext
    {
        public PaintingAnalyzerDbContext(DbContextOptions<PaintingAnalyzerDbContext> options): base(options)
        {

        }

        public DbSet<ImageModel> Images { get; set; }

        public DbSet<UserModel> Users { get; set; }

        public DbSet<NotificationModel> Notifications { get; set; }

        public DbSet<AnalyzeModel> Analyzes { get; set; }

        public DbSet<Session_Artist_AccuracyModel> Session_Artist_Accuracy { get; set; }

        public DbSet<Session_Genre_AccuracyModel> Session_Genre_Accuracy { get; set; }

        public DbSet<Session_Style_AccuracyModel> Session_Style_Accuracy { get; set; }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {



            modelBuilder.Entity<ImageModel>()
              .HasOne(p => p.User)
              .WithMany(b => b.UploadeImages)
              .OnDelete(DeleteBehavior.Cascade);


            modelBuilder.Entity<AnalyzeModel>()
                .HasOne(p => p.Image)
                .WithMany(b => b.Analyzes)
                .OnDelete(DeleteBehavior.Cascade);

            modelBuilder.Entity<Session_Artist_AccuracyModel>()
                .HasOne(p => p.Analyze)
                .WithMany(b => b.Session_Artist_Accuracy)
                .OnDelete(DeleteBehavior.Cascade);
            
            modelBuilder.Entity<Session_Genre_AccuracyModel>()
                .HasOne(p => p.Analyze)
                .WithMany(b => b.Session_Genre_Accuracy)
                .OnDelete(DeleteBehavior.Cascade);
            
            modelBuilder.Entity<Session_Style_AccuracyModel>()
                .HasOne(p => p.Analyze)
                .WithMany(b => b.Session_Style_Accuracy)
                .OnDelete(DeleteBehavior.Cascade);

        }


    }
}
