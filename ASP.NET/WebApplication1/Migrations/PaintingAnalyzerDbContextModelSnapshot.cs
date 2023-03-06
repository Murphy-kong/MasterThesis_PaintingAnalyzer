﻿// <auto-generated />
using System;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Infrastructure;
using Microsoft.EntityFrameworkCore.Metadata;
using Microsoft.EntityFrameworkCore.Storage.ValueConversion;
using WebApplication1.Models;

#nullable disable

namespace WebApplication1.Migrations
{
    [DbContext(typeof(PaintingAnalyzerDbContext))]
    partial class PaintingAnalyzerDbContextModelSnapshot : ModelSnapshot
    {
        protected override void BuildModel(ModelBuilder modelBuilder)
        {
#pragma warning disable 612, 618
            modelBuilder
                .HasAnnotation("ProductVersion", "6.0.8")
                .HasAnnotation("Relational:MaxIdentifierLength", 128);

            SqlServerModelBuilderExtensions.UseIdentityColumns(modelBuilder, 1L, 1);

            modelBuilder.Entity("NotificationModelUserModel", b =>
                {
                    b.Property<int>("NotificationsNotificationID")
                        .HasColumnType("int");

                    b.Property<int>("UsersUserID")
                        .HasColumnType("int");

                    b.HasKey("NotificationsNotificationID", "UsersUserID");

                    b.HasIndex("UsersUserID");

                    b.ToTable("NotificationModelUserModel");
                });

            modelBuilder.Entity("WebApplication1.Models.AnalyzeModel", b =>
                {
                    b.Property<int?>("AnalyzeID")
                        .ValueGeneratedOnAdd()
                        .HasColumnType("int");

                    SqlServerPropertyBuilderExtensions.UseIdentityColumn(b.Property<int?>("AnalyzeID"), 1L, 1);

                    b.Property<int?>("ImageID")
                        .HasColumnType("int");

                    b.Property<string>("ML_Model")
                        .HasColumnType("nvarchar(max)");

                    b.Property<DateTime>("ReleaseDate")
                        .HasColumnType("datetime2");

                    b.HasKey("AnalyzeID");

                    b.HasIndex("ImageID");

                    b.ToTable("Analyzes");
                });

            modelBuilder.Entity("WebApplication1.Models.ImageModel", b =>
                {
                    b.Property<int>("ImageID")
                        .ValueGeneratedOnAdd()
                        .HasColumnType("int");

                    SqlServerPropertyBuilderExtensions.UseIdentityColumn(b.Property<int>("ImageID"), 1L, 1);

                    b.Property<string>("ImageName")
                        .IsRequired()
                        .HasColumnType("nvarchar(100)");

                    b.Property<string>("Occupation")
                        .IsRequired()
                        .HasColumnType("nvarchar(50)");

                    b.Property<DateTime>("ReleaseDate")
                        .HasColumnType("datetime2");

                    b.Property<int?>("UserID")
                        .IsRequired()
                        .HasColumnType("int");

                    b.HasKey("ImageID");

                    b.HasIndex("UserID");

                    b.ToTable("Images");
                });

            modelBuilder.Entity("WebApplication1.Models.NotificationModel", b =>
                {
                    b.Property<int>("NotificationID")
                        .ValueGeneratedOnAdd()
                        .HasColumnType("int");

                    SqlServerPropertyBuilderExtensions.UseIdentityColumn(b.Property<int>("NotificationID"), 1L, 1);

                    b.Property<string>("Content")
                        .IsRequired()
                        .HasColumnType("nvarchar(max)");

                    b.Property<DateTime>("ReleaseDate")
                        .HasColumnType("datetime2");

                    b.Property<string>("Type")
                        .IsRequired()
                        .HasColumnType("nvarchar(max)");

                    b.HasKey("NotificationID");

                    b.ToTable("Notifications");
                });

            modelBuilder.Entity("WebApplication1.Models.Session_Artist_AccuracyModel", b =>
                {
                    b.Property<int>("ArtistID")
                        .ValueGeneratedOnAdd()
                        .HasColumnType("int");

                    SqlServerPropertyBuilderExtensions.UseIdentityColumn(b.Property<int>("ArtistID"), 1L, 1);

                    b.Property<float>("Accuracy")
                        .HasColumnType("real");

                    b.Property<int?>("AnalyzeID")
                        .HasColumnType("int");

                    b.Property<string>("ArtistName")
                        .IsRequired()
                        .HasColumnType("nvarchar(max)");

                    b.HasKey("ArtistID");

                    b.HasIndex("AnalyzeID");

                    b.ToTable("Session_Artist_Accuracy");
                });

            modelBuilder.Entity("WebApplication1.Models.Session_Genre_AccuracyModel", b =>
                {
                    b.Property<int>("GenreID")
                        .ValueGeneratedOnAdd()
                        .HasColumnType("int");

                    SqlServerPropertyBuilderExtensions.UseIdentityColumn(b.Property<int>("GenreID"), 1L, 1);

                    b.Property<float>("Accuracy")
                        .HasColumnType("real");

                    b.Property<int?>("AnalyzeID")
                        .HasColumnType("int");

                    b.Property<string>("GenreName")
                        .IsRequired()
                        .HasColumnType("nvarchar(max)");

                    b.HasKey("GenreID");

                    b.HasIndex("AnalyzeID");

                    b.ToTable("Session_Genre_Accuracy");
                });

            modelBuilder.Entity("WebApplication1.Models.Session_Style_AccuracyModel", b =>
                {
                    b.Property<int>("StyleID")
                        .ValueGeneratedOnAdd()
                        .HasColumnType("int");

                    SqlServerPropertyBuilderExtensions.UseIdentityColumn(b.Property<int>("StyleID"), 1L, 1);

                    b.Property<float>("Accuracy")
                        .HasColumnType("real");

                    b.Property<int?>("AnalyzeID")
                        .HasColumnType("int");

                    b.Property<string>("StyleName")
                        .IsRequired()
                        .HasColumnType("nvarchar(max)");

                    b.HasKey("StyleID");

                    b.HasIndex("AnalyzeID");

                    b.ToTable("Session_Style_Accuracy");
                });

            modelBuilder.Entity("WebApplication1.Models.UserModel", b =>
                {
                    b.Property<int>("UserID")
                        .ValueGeneratedOnAdd()
                        .HasColumnType("int");

                    SqlServerPropertyBuilderExtensions.UseIdentityColumn(b.Property<int>("UserID"), 1L, 1);

                    b.Property<string>("Email")
                        .HasColumnType("nvarchar(max)");

                    b.Property<byte[]>("PasswordHash")
                        .IsRequired()
                        .HasColumnType("varbinary(max)");

                    b.Property<byte[]>("PasswordSalt")
                        .IsRequired()
                        .HasColumnType("varbinary(max)");

                    b.Property<DateTime>("ReleaseDate")
                        .HasColumnType("datetime2");

                    b.Property<string>("Role")
                        .IsRequired()
                        .HasColumnType("nvarchar(max)");

                    b.Property<string>("UserName")
                        .HasColumnType("nvarchar(max)");

                    b.HasKey("UserID");

                    b.ToTable("Users");
                });

            modelBuilder.Entity("NotificationModelUserModel", b =>
                {
                    b.HasOne("WebApplication1.Models.NotificationModel", null)
                        .WithMany()
                        .HasForeignKey("NotificationsNotificationID")
                        .OnDelete(DeleteBehavior.Cascade)
                        .IsRequired();

                    b.HasOne("WebApplication1.Models.UserModel", null)
                        .WithMany()
                        .HasForeignKey("UsersUserID")
                        .OnDelete(DeleteBehavior.Cascade)
                        .IsRequired();
                });

            modelBuilder.Entity("WebApplication1.Models.AnalyzeModel", b =>
                {
                    b.HasOne("WebApplication1.Models.ImageModel", "Image")
                        .WithMany("Analyzes")
                        .HasForeignKey("ImageID")
                        .OnDelete(DeleteBehavior.Cascade);

                    b.Navigation("Image");
                });

            modelBuilder.Entity("WebApplication1.Models.ImageModel", b =>
                {
                    b.HasOne("WebApplication1.Models.UserModel", "User")
                        .WithMany("UploadeImages")
                        .HasForeignKey("UserID")
                        .OnDelete(DeleteBehavior.Cascade)
                        .IsRequired();

                    b.Navigation("User");
                });

            modelBuilder.Entity("WebApplication1.Models.Session_Artist_AccuracyModel", b =>
                {
                    b.HasOne("WebApplication1.Models.AnalyzeModel", "Analyze")
                        .WithMany("Session_Artist_Accuracy")
                        .HasForeignKey("AnalyzeID")
                        .OnDelete(DeleteBehavior.Cascade);

                    b.Navigation("Analyze");
                });

            modelBuilder.Entity("WebApplication1.Models.Session_Genre_AccuracyModel", b =>
                {
                    b.HasOne("WebApplication1.Models.AnalyzeModel", "Analyze")
                        .WithMany("Session_Genre_Accuracy")
                        .HasForeignKey("AnalyzeID")
                        .OnDelete(DeleteBehavior.Cascade);

                    b.Navigation("Analyze");
                });

            modelBuilder.Entity("WebApplication1.Models.Session_Style_AccuracyModel", b =>
                {
                    b.HasOne("WebApplication1.Models.AnalyzeModel", "Analyze")
                        .WithMany("Session_Style_Accuracy")
                        .HasForeignKey("AnalyzeID")
                        .OnDelete(DeleteBehavior.Cascade);

                    b.Navigation("Analyze");
                });

            modelBuilder.Entity("WebApplication1.Models.AnalyzeModel", b =>
                {
                    b.Navigation("Session_Artist_Accuracy");

                    b.Navigation("Session_Genre_Accuracy");

                    b.Navigation("Session_Style_Accuracy");
                });

            modelBuilder.Entity("WebApplication1.Models.ImageModel", b =>
                {
                    b.Navigation("Analyzes");
                });

            modelBuilder.Entity("WebApplication1.Models.UserModel", b =>
                {
                    b.Navigation("UploadeImages");
                });
#pragma warning restore 612, 618
        }
    }
}